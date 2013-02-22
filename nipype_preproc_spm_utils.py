"""
:Module: nipype_preproc_spm_utils
:Synopsis: routine functions for SPM preprocessing business
:Author: dohmatob elvis dopgima (hereafter referred to as DED)

XXX TODO: document the code!
XXX TODO: re-factor the code!
"""

# standard imports
import os
import shutil
import json

# imports for caching (yeah, we aint got time to loose!)
from nipype.caching import Memory

# imports i/o
from nipype.interfaces.base import Bunch
from io_utils import delete_orientation, is_3D, get_vox_dims,\
    resample_img, do_3Dto4D_merge, compute_mean_3D_image, compute_mean_image

# spm and matlab imports
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab

# parallelism imports
import joblib

# find package path
root_dir = os.path.split(os.path.abspath(__file__))[0]

# set job count
N_JOBS = -1
if 'N_JOBS' in os.environ:
    N_JOBS = int(os.environ['N_JOBS'])

# set matlab exec path
MATLAB_EXEC = "/neurospin/local/matlab/bin/matlab"
if 'MATLAB_EXEC' in os.environ:
    MATLAB_EXEC = os.environ['MATLAB_EXEC']
assert os.path.exists(MATLAB_EXEC), \
    "nipype_preproc_smp_utils: MATLAB_EXEC: %s, \
doesn't exist; you need to export MATLAB_EXEC" % MATLAB_EXEC
matlab.MatlabCommand.set_default_matlab_cmd(MATLAB_EXEC)

# set matlab SPM back-end path
SPM_DIR = '/i2bm/local/spm8'
if 'SPM_DIR' in os.environ:
    SPM_DIR = os.environ['SPM_DIR']
assert os.path.exists(SPM_DIR), \
    "nipype_preproc_smp_utils: SPM_DIR: %s,\
 doesn't exist; you need to export SPM_DIR" % SPM_DIR
matlab.MatlabCommand.set_default_paths(SPM_DIR)

# set templates
EPI_TEMPLATE = os.path.join(SPM_DIR, 'templates/EPI.nii')
T1_TEMPLATE = "/usr/share/data/fsl-mni152-templates/avg152T1.nii"
if not os.path.isfile(T1_TEMPLATE):
    T1_TEMPLATE += '.gz'
    if not os.path.exists(T1_TEMPLATE):
        T1_TEMPLATE = os.path.join(SPM_DIR, "templates/T1.nii")

#os.path.join(SPM_DIR, 'templates/T1.nii')
GM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/grey.nii')
WM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/white.nii')
CSF_TEMPLATE = os.path.join(SPM_DIR, 'tpm/csf.nii')

# MISC
SPM8_URL = "http://www.fil.ion.ucl.ac.uk/spm/software/spm8/"
PYPREPROCESS_URL = "https://github.com/neurospin/pypreprocess"
DARTEL_URL = ("http://www.fil.ion.ucl.ac.uk/spm/software/spm8/"
              "SPM8_Release_Notes.pdf")
NIPYPE_URL = "http://nipy.sourceforge.net/nipype/"


class SubjectData(Bunch):
    """
    Encapsulation for subject data, relative to preprocessing.

    XXX Use custom objects (dicts, tuples, etc.) instead of this 'Bunch' stuff.

    """

    def __init__(self):
        self.subject_id = "subXYZ"
        self.session_id = ["UNKNOWN_SESSION"]
        self.anat = None
        self.func = None
        self.bad_orientation = False

    def delete_orientation(self):
        # prepare for smart caching
        cache_dir = os.path.join(self.output_dir, 'deleteorient_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        mem = joblib.Memory(cachedir=cache_dir, verbose=5)

        # deleteorient for func
        self.func = [mem.cache(delete_orientation)(
                self.func[j],
                self.output_dir,
                output_tag=self.session_id[j])
                     for j in xrange(len(self.session_id))]

        # deleteorient for anat
        if not self.anat is None:
            self.anat = mem.cache(delete_orientation)(
                self.anat, self.output_dir)

    def sanitize(self, do_deleteorient=False):
        if type(self.session_id) is str:
            self.session_id = [self.session_id]
        if type(self.func) is str:
            self.func = [self.func]
        if is_3D(self.func[0]):
            self.func = [self.func]

        assert len(self.func) == len(self.session_id)

        if do_deleteorient or self.bad_orientation:
            self.delete_orientation()


def generate_normalization_thumbnails(
    normalized_files,
    output_dir,
    brain="epi",
    cmap=None,
    results_gallery=None,
    progress_logger=None,
    ):
    """Generate thumbnails after spatial normalization or subject

    Parameters
    ----------
    normalized_files: list
        paths to normalized images (3Ds or 4Ds)

    output_dir: string
        dir to which all output will be written

    brain: string (optional)
        a short comment/tag like 'epi', or 'anat'

    cmap: optional
        cmap (color map) to use for plots

    result_gallery: reporter.ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

    import check_preprocessing
    import reporting.reporter as reporter

    _brain = brain

    if type(normalized_files) is str:
        nipype_report_filename = os.path.join(
            os.path.dirname(normalized_files),
            "_report/report.rst")
        normalized = normalized_files
    else:
        brain = "mean" + brain

        nipype_report_filename = os.path.join(
            os.path.dirname(normalized_files[0]),
            "_report/report.rst")

        mean_normalized_img = compute_mean_3D_image(normalized_files)
        normalized = mean_normalized_img

    # nipype report
    nipype_html_report_filename = os.path.join(
        output_dir,
        '%s_normalize_nipype_report.html' % brain)
    nipype_report = reporter.nipype2htmlreport(
        nipype_report_filename)
    open(nipype_html_report_filename,
                 'w').write(str(nipype_report))

    if progress_logger:
        progress_logger.log(nipype_report.split('Terminal output')[0])
        progress_logger.log('<hr/>')

    output = {}

    # prepare for smart caching
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    # plot outline (edge map) of SPM MNI template on the
    # normalized image
    target = T1_TEMPLATE
    source = normalized

    outline = os.path.join(
        output_dir,
        "%s_on_%s_outline.png" % (os.path.basename(target),
                                  brain))

    qa_mem.cache(check_preprocessing.plot_registration)(
        target,
        source,
        output_filename=outline,
        cmap=cmap,
        title="Outline of SPM MNI %s template on %s" % (
            os.path.basename(target),
            brain))

    # create thumbnail
    if results_gallery:
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(href=os.path.basename(outline))
        thumbnail.img = reporter.img(
            src=os.path.basename(outline), height="250px")
        thumbnail.description = \
            "Normalization of %s (<a href=%s>see execution log</a>)" \
            % (_brain, os.path.basename(nipype_html_report_filename))

        results_gallery.commit_thumbnails(thumbnail)

    # plot outline (edge map) of the normalized image
    # on the SPM MNI template
    source, target = (target, source)
    outline = os.path.join(
        output_dir,
        "%s_on_%s_outline.png" % (brain,
                                  os.path.basename(source)))
    outline_axial = os.path.join(
        output_dir,
        "%s_on_%s_outline_axial.png" % (brain,
                                        os.path.basename(source)))

    qa_mem.cache(check_preprocessing.plot_registration)(
        target,
        source,
        output_filename=outline_axial,
        slicer='z',
        cmap=cmap,
        title="Outline of %s on SPM MNI %s template" % (
            brain,
            os.path.basename(source)))

    output['axial'] = outline_axial

    qa_mem.cache(check_preprocessing.plot_registration)(
        target,
        source,
        output_filename=outline,
        cmap=cmap,
        title="Outline of %s on MNI %s template" % (
            brain,
            os.path.basename(source)))

    # create thumbnail
    if results_gallery:
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(href=os.path.basename(outline))
        thumbnail.img = reporter.img(
            src=os.path.basename(outline), height="250px")
        thumbnail.description = \
            "Normalization of %s (<a href=%s>see execution log</a>)" \
            % (_brain, os.path.basename(nipype_html_report_filename))

        results_gallery.commit_thumbnails(thumbnail)

    return output


def generate_segmentation_thumbnails(
    normalized_files,
    output_dir,
    subject_gm_file=None,
    subject_wm_file=None,
    subject_csf_file=None,
    brain='epi',
    cmap=None,
    results_gallery=None,
    progress_logger=None):
    """Generates thumbnails after indirect normalization
    (segmentation + normalization)

    Parameters
    ----------
    normalized_file: list
        paths to normalized images (3Ds or 4Ds)

    output_dir: string
        dir to which all output will be written

    subject_gm_file: string (optional)
        path to subject GM file

    subject_csf_file: string (optional)
        path to subject WM file

    subject_csf_file: string (optional)
        path to subject CSF file

    brain: string (optional)
        a short commeent/tag like 'epi', or 'anat'

    cmap: optional
        cmap (color map) to use for plots

    result_gallery: reporter.ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

    import check_preprocessing
    import reporting.reporter as reporter

    _brain = brain

    if progress_logger:
        progress_logger.log('<b>Normalization of %s</b><br/><br/>' % _brain)

    if type(normalized_files) is str:
        nipype_report_filename = os.path.join(
            os.path.dirname(normalized_files),
            "_report/report.rst")
        normalized_file = normalized_files
    else:
        brain = "mean" + brain

        nipype_report_filename = os.path.join(
            os.path.dirname(normalized_files[0]),
            "_report/report.rst")

        mean_normalized_file = os.path.join(
            os.path.dirname(normalized_files[0]),
            "mean%s.nii" % brain)

        compute_mean_3D_image(normalized_files,
                           output_filename=mean_normalized_file)
        normalized_file = mean_normalized_file

    # nipype report
    nipype_html_report_filename = os.path.join(
        output_dir,
        '%s_normalize_nipype_report.html' % brain)
    nipype_report = reporter.nipype2htmlreport(
        nipype_report_filename)
    open(nipype_html_report_filename,
                 'w').write(str(nipype_report))
    output = {}

    # prepare for smart caching
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    # plot contours of template compartments on subject's brain
    template_compartments_contours = os.path.join(
        output_dir,
        "template_tmps_contours_on_%s.png" % brain)
    template_compartments_contours_axial = os.path.join(
        output_dir,
        "template_compartments_contours_on_%s_axial.png" % brain)

    qa_mem.cache(check_preprocessing.plot_segmentation)(
        normalized_file,
        GM_TEMPLATE,
        wm_filename=WM_TEMPLATE,
        csf_filename=CSF_TEMPLATE,
        output_filename=template_compartments_contours_axial,
        slicer='z',
        cmap=cmap,
        title="template TPMs")

    qa_mem.cache(check_preprocessing.plot_segmentation)(
        normalized_file,
        gm_filename=GM_TEMPLATE,
        wm_filename=WM_TEMPLATE,
        csf_filename=CSF_TEMPLATE,
        output_filename=template_compartments_contours,
        cmap=cmap,
        title=("Template GM, WM, and CSF contours on "
               "subject's %s") % brain)

    # create thumbnail
    if results_gallery:
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(
            href=os.path.basename(template_compartments_contours))
        thumbnail.img = reporter.img(
            src=os.path.basename(template_compartments_contours),
            height="250px")
        thumbnail.description = (
            "Normalization of %s (<a href=%s>see "
            "execution log</a>)") % (_brain, os.path.basename(
            nipype_html_report_filename))

        results_gallery.commit_thumbnails(thumbnail)

    # plot contours of subject's compartments on subject's brain
    if subject_gm_file:
        subject_compartments_contours = os.path.join(
            output_dir,
            "subject_tmps_contours_on_subject_%s.png" % brain)
        subject_compartments_contours_axial = os.path.join(
            output_dir,
            "subject_tmps_contours_on_subject_%s_axial.png" % brain)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            normalized_file,
            subject_gm_file,
            wm_filename=subject_wm_file,
            csf_filename=subject_csf_file,
            output_filename=subject_compartments_contours_axial,
            slicer='z',
            cmap=cmap,
            title="subject TPMs")

        title_prefix = "Subject's GM"
        if subject_wm_file:
            title_prefix += ", WM"
        if subject_csf_file:
            title_prefix += ", and CSF"
        qa_mem.cache(check_preprocessing.plot_segmentation)(
            normalized_file,
            subject_gm_file,
            wm_filename=subject_wm_file,
            csf_filename=subject_csf_file,
            output_filename=subject_compartments_contours,
            cmap=cmap,
            title=("%s contours on "
               "subject's %s") % (title_prefix, brain))

        # create thumbnail
        if results_gallery:
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(
                href=os.path.basename(subject_compartments_contours))
            thumbnail.img = reporter.img(
                src=os.path.basename(subject_compartments_contours),
                height="250px")
            thumbnail.description = \
                "Normalization of %s (<a href=%s>see execution log</a>)" \
                % (_brain, os.path.basename(nipype_html_report_filename))

            results_gallery.commit_thumbnails(thumbnail)

    output['axials'] = {}
    output['axial'] = template_compartments_contours_axial

    return output


def commit_subject_thumnbail_to_parent_gallery(
    thumbnail,
    subject_id,
    parent_results_gallery):
    """Commit thumbnail (summary of subject_report) to parent results gallery,
    correcting attrs of the embedded img object as necessary.

    Parameters
    ----------
    thumbnail: reporter.Thumbnail instance
        thumbnail to be committed

    subject_id: string
        subject_id for subject under inspection

    result_gallery: reporter.ResultsGallery instance (optional)
        gallery to which thumbnail will be committed

    """

    thumbnail.img.height = "250px"
    thumbnail.img.src = "%s/%s" % (
        subject_id,
        os.path.basename(thumbnail.img.src))
    thumbnail.a.href = "%s/%s" % (
        subject_id,
        os.path.basename(thumbnail.a.href))
    parent_results_gallery.commit_thumbnails(thumbnail)


def generate_cv_tc_thumbnail(
    image_files,
    sessions,
    subject_id,
    output_dir,
    plot_diff=True,
    results_gallery=None):
    """Generate cv tc thumbnails

    Parameters
    ----------
    image_files: list or strings or list
        paths (4D case) to list of paths (3D case) of images under inspection

    output_dir: string
        dir to which all output whill be written

    subject_id: string
        id of subject under inspection

    sessions: list
        list of session ids, one per element of image_files

    result_gallery: reporter.ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

    import reporting.reporter as reporter
    import check_preprocessing

    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    if type(image_files) is str:
        image_files = [image_files]
    else:
        if is_3D(image_files[0]):
            image_files = [image_files]

    assert len(sessions) == len(image_files)

    cv_tc_plot_output_file = os.path.join(
        output_dir,
        "cv_tc_plot.png")

    qa_mem.cache(
        check_preprocessing.plot_cv_tc)(
        image_files,
        sessions,
        subject_id,
        cv_tc_plot_outfile=cv_tc_plot_output_file,
        plot_diff=True)

    # create thumbnail
    thumbnail = reporter.Thumbnail()
    thumbnail.a = reporter.a(
        href=os.path.basename(cv_tc_plot_output_file))
    thumbnail.img = reporter.img(
        src=os.path.basename(cv_tc_plot_output_file), height="250px",
        width="600px")
    thumbnail.description = "Coefficient of Variation (%d sessions)"\
                                 % len(sessions)

    if results_gallery:
        results_gallery.commit_thumbnails(thumbnail)


def _do_subject_realign(output_dir,
                        subject_id=None,
                        sessions=[1],
                        do_report=True,
                        results_gallery=None,
                        progress_logger=None,
                       **spm_realign_kwargs):
    """
    Wrapper for nipype.interfaces.spm.Realign.

    Does realignment and generates QA plots (motion parameters, etc.).

    Parameters
    ----------
    output_dir: string
        An existing folder where all output files will be written.

    subject_id: string (optional)
        id of the subject being preprocessed

    do_report: boolean (optional)
        if true, then QA plots will be generated after executing the realign
        node.

    *spm_realign_kwargs: kwargs (paramete-value dict)
        parameters to be passed to the nipype.interfaces.spm.Realign back-end
        node

    """

    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    if progress_logger:
        progress_logger.log('<b>Motion Correction</b><br/><br/>')

    # run workflow
    realign = mem.cache(spm.Realign)
    realign_result = realign(**spm_realign_kwargs)

    # generate gallery for HTML report
    if do_report:
        import check_preprocessing
        import reporting.reporter as reporter

        rp_files = realign_result.outputs.realignment_parameters
        if type(rp_files) is str:
            rp_files = [rp_files]

        assert len(sessions) == len(rp_files), rp_files

        # nipype report
        nipype_report_filename = os.path.join(
            os.path.dirname(rp_files[0]),
            "_report/report.rst")
        nipype_html_report_filename = os.path.join(
            output_dir,
            'realign_nipype_report.html')
        nipype_report = reporter.nipype2htmlreport(nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        if progress_logger:
            progress_logger.log(nipype_report.split('Terminal output')[0])
            progress_logger.log('<hr/>')

        for session_id, rp in zip(sessions, rp_files):
            rp_plot = os.path.join(
                output_dir, 'rp_plot_%s.png' % session_id)
            check_preprocessing.plot_spm_motion_parameters(
                rp,
                title="Plot of Estimated motion for session %s" % session_id,
                output_filename=rp_plot)

            # create thumbnail
            if results_gallery:
                thumbnail = reporter.Thumbnail()
                thumbnail.a = reporter.a(href=os.path.basename(rp_plot))
                thumbnail.img = reporter.img(src=os.path.basename(rp_plot),
                                             height="250px",
                                             width="600px")
                thumbnail.description = \
                    "Motion Correction (<a href=%s>see execution log</a>)" % \
                    os.path.basename(nipype_html_report_filename)

                results_gallery.commit_thumbnails(thumbnail)

        output['rp_plot'] = rp_plot

    # collect ouput
    output['result'] = realign_result

    return output


def _do_subject_coreg(output_dir,
                      subject_id=None,
                      do_report=True,
                      results_gallery=None,
                      progress_logger=None,
                      coreg_func_to_anat=False,
                      comments="",
                      **spm_coregister_kwargs):
    """
    Wrapper for nipype.interfaces.spm.Coregister.

    Does coregistration and generates QA plots (outline of coregistered source
    on target, etc.).

    Parameters
    ----------
    output_dir: string
        An existing folder where all output files will be written.

    subject_id: string (optional)
        id of the subject being preprocessed

    do_report: boolean (optional)
        if true, then QA plots will be generated after executing the coregister
        node.

    *spm_coregister_kwargs: kwargs (paramete-value dict)
        parameters to be passed to the nipype.interfaces.spm.Coregister
        back-end node

    """

    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    if progress_logger:
        progress_logger.log('<b>Coregistration</b><br/><br/>')

    # run workflow
    coreg = mem.cache(spm.Coregister)
    coreg_result = coreg(**spm_coregister_kwargs)

    # generate gallery for HTML report
    if do_report:
        import check_preprocessing
        import reporting.reporter as reporter
        import pylab as pl

        # nipype report
        nipype_report_filename = os.path.join(
            os.path.dirname(coreg_result.outputs.coregistered_source),
            "_report/report.rst")
        nipype_html_report_filename = os.path.join(
            output_dir,
            'coregister_nipype_report.html')
        nipype_report = reporter.nipype2htmlreport(nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        if progress_logger:
            progress_logger.log(nipype_report.split('Terminal output')[0])
            progress_logger.log('<hr/>')

        # prepare for smart caching
        qa_cache_dir = os.path.join(output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

        # plot outline of target on coregistered source
        target = spm_coregister_kwargs['target']
        source = coreg_result.outputs.coregistered_source

        outline = os.path.join(
            output_dir,
            "%s_on_%s_outline.png" % (os.path.basename(target),
                                      os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline,
            cmap=pl.cm.gray,
            title="Outline of %s on %s" % (os.path.basename(target),
                                           os.path.basename(source)))

        outline_axial = os.path.join(
            output_dir,
            "%s_on_%s_outline_axial.png" % (os.path.basename(target),
                                            os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline_axial,
            slicer='z',
            cmap=pl.cm.gray,
            title="%s: coreg" % subject_id)

        # create thumbnail
        if results_gallery:
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(href=os.path.basename(outline))
            thumbnail.img = reporter.img(src=os.path.basename(outline),
                                         height="250px")
            thumbnail.description = \
                "Coregistration %s (<a href=%s>see execution log</a>)" % \
                (comments, os.path.basename(nipype_html_report_filename))

            results_gallery.commit_thumbnails(thumbnail)

        output['axial_outline'] = outline_axial

        # plot outline of coregistered source on target
        source, target = (target, source)
        outline = os.path.join(
            output_dir,
            "%s_on_%s_outline.png" % (os.path.basename(target),
                                      os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline,
            cmap=pl.cm.gray,
            title="Outline of %s on %s" % (os.path.basename(target),
                                           os.path.basename(source)))

        outline_axial = os.path.join(
            output_dir,
            "%s_on_%s_outline_axial.png" % (os.path.basename(target),
                                            os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline_axial,
            cmap=pl.cm.gray,
            slicer='z',
            title="%s: coreg" % subject_id)

        # create thumbnail
        if results_gallery:
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(href=os.path.basename(outline))
            thumbnail.img = reporter.img(src=os.path.basename(outline),
                                         height="250px")
            thumbnail.description = \
                "Coregistration %s (<a href=%s>see execution log</a>)" \
                % (comments, os.path.basename(nipype_html_report_filename))
            results_gallery.commit_thumbnails(thumbnail)

    # collect ouput
    output['result'] = coreg_result

    return output


def _do_subject_segment(output_dir,
                       subject_id=None,
                       do_report=True,
                        progress_logger=None,
                       **spm_segment_kwargs):
    """
    Wrapper for nipype.interfaces.spm.Segment.

    Does segmentation of brain into GM, WM, and CSF compartments and
    generates QA plots.

    Parameters
    ----------
    output_dir: string
        An existing folder where all output files will be written.

    subject_id: string (optional)
        id of the subject being preprocessed

    do_report: boolean (optional)
        if true, then QA plots will be generated after executing the segment
        node.

    *spm_segment_kwargs: kwargs (paramete-value dict)
        parameters to be passed to the nipype.interfaces.spm.Segment back-end
        node

    """

    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    if progress_logger:
        progress_logger.log('<b>Segmentation</b><br/><br/>')

    # run workflow
    segment = mem.cache(spm.Segment)
    segment_result = segment(**spm_segment_kwargs)

    # generate gallery for HTML report
    if do_report:
        import reporting.reporter as reporter

        # nipype report
        nipype_report_filename = os.path.join(
            os.path.dirname(segment_result.outputs.transformation_mat),
            "_report/report.rst")
        nipype_html_report_filename = os.path.join(
            output_dir,
            'coregister_nipype_report.html')
        nipype_report = reporter.nipype2htmlreport(nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        if progress_logger:
            progress_logger.log(nipype_report.split('Terminal output')[0])
            progress_logger.log('<hr/>')

    # collect ouput
    output['result'] = segment_result

    return output


def _do_subject_normalize(output_dir,
                          subject_id=None,
                          sessions=[1],
                          segment_result=None,
                          do_report=True,
                          results_gallery=None,
                          progress_logger=None,
                          brain="epi",
                          cmap=None,
                          fwhm=0,
                          **spm_normalize_kwargs):
    """
    Wrapper for nipype.interfaces.spm.Normalize.

    Does normalization and generates QA plots (outlines of normalized files on
    template, etc.).

    Parameters
    ----------
    output_dir: string
        An existing folder where all output files will be written.

    subject_id: string (optional)
        id of the subject being preprocessed

    do_report: boolean (optional)
        if true, then QA plots will be generated after executing the normalize
        node.

    *spm_normalize_kwargs: kwargs (paramete-value dict)
        parameters to be passed to the nipype.interfaces.spm.Normalize back-end
        node

    """

    output = {}

    if progress_logger:
        progress_logger.log('<b>Normalization of %s</b><br/><br/>' % brain)

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    # run workflow
    normalize = mem.cache(spm.Normalize)
    norm_result = normalize(**spm_normalize_kwargs)
    normalized_files = norm_result.outputs.normalized_files

    # collect ouput
    output['result'] = norm_result
    if not norm_result.outputs is None:
        normalized_files = norm_result.outputs.normalized_files
        output['normalized_files'] = normalized_files

    if fwhm:
        print fwhm
        smooth = mem.cache(spm.Smooth)
        smooth_result = smooth(
            in_files=normalized_files,
            fwhm=fwhm)

        # collect ouput
        output['result'] = smooth_result
        if not smooth_result.outputs is None:
            normalized_files = smooth_result.outputs.smoothed_files
            output['normalized_files'] = normalized_files

    # generate gallery for HTML report
    if do_report:
        # generate normalization thumbs
        output.update(generate_normalization_thumbnails(
                normalized_files,
                output_dir,
                brain=brain,
                cmap=cmap,
                results_gallery=results_gallery,
                progress_logger=progress_logger))

        # generate segmentation thumbs
        if segment_result:
            subject_gm_file = segment_result.outputs.modulated_gm_image
            subject_wm_file = segment_result.outputs.modulated_wm_image
            subject_csf_file = segment_result.outputs.modulated_csf_image
        else:
            subject_gm_file = None
            subject_wm_file = None
            subject_csf_file = None
        output.update(generate_segmentation_thumbnails(
                normalized_files,
                output_dir,
                subject_gm_file=subject_gm_file,
                subject_wm_file=subject_wm_file,
                subject_csf_file=subject_csf_file,
                brain=brain,
                cmap=cmap,
                results_gallery=results_gallery))

    # collect ouput
    output['result'] = norm_result

    return output


def _do_subject_preproc(
    subject_data,
    do_deleteorient=False,
    do_report=True,
    fwhm=0,
    do_bet=False,
    do_slicetiming=False,
    do_realign=True,
    do_coreg=True,
    func_to_anat=False,
    do_segment=True,
    do_normalize=True,
    do_cv_tc=True,
    additional_preproc_undergone=None,
    parent_results_gallery=None,
    subject_progress_logger=None,
    last_stage=True,
    main_page="#",
    ignore_exception=True):
    """
    Function preprocessing data for a single subject.

    Parameters
    ----------
    subject_data: instance of SubjectData
        Object containing information about the subject under inspection
        (path to anat image, func image(s),
        output directory, etc.)

    delete_orientation: bool (optional)
        if true, then orientation meta-data in all input image files for this
        subject will be stripped-off

    do_report: bool (optional)
        if set, post-preprocessing QA report will be generated.

    do_bet: bool (optional)
        if set, brain-extraction will be applied to remove non-brain tissue
        before preprocessing (this can help prevent the scull from aligning
        with the cortical surface, for example)

    do_slicetiming: bool (optional)
        if set, slice-timing correct temporal mis-alignment of the functional
        slices

    do_realign: bool (optional)
        if set, then the functional data will be realigned to correct for
        head-motion

    do_coreg: bool (optional)
        if set, then subject anat image (of the spm EPI template, if the later
        if not available) will be coregistered against the functional images
        (i.e the mean thereof)

    do_segment: bool (optional)
        if set, then the subject's anat image will be segmented to produce GM,
        WM, and CSF compartments (useful for both indirect normalization
        (intra-subject) or DARTEL (inter-subject) alike


    do_cv_tc: bool (optional)
        if set, a summarizing the time-course of the coefficient of variation
        in the preprocessed fMRI time-series will be generated

    parent_results_gallery: reporter.ResulsGallery object (optional)
        a handle to the results gallery to which the final QA thumail for this
        subject will be committed

     main_page: string (optional)
        the href to the QA report main page

    """

    # sanity
    subject_data.sanitize(do_deleteorient=do_deleteorient)

    output = {"subject_id": subject_data.subject_id,
              "session_id": subject_data.session_id,
              "output_dir": subject_data.output_dir}

    # create subject_data.output_dir if dir doesn't exist
    if not os.path.exists(subject_data.output_dir):
        os.makedirs(subject_data.output_dir)

    if do_report:
        import time
        import reporting.reporter as reporter

        # compute docstring explaining preproc steps undergone
        preproc_undergone = """\
<p>All preprocessing has been done using <a href="%s">pypreprocess</a>,
 which is powered by <a href="%s">nipype</a>, and <a href="%s">SPM8</a>.
</p>""" % (PYPREPROCESS_URL, NIPYPE_URL, SPM8_URL)

        preproc_undergone += "<ul>"

        if do_deleteorient or subject_data.bad_orientation:
            preproc_undergone += (
                "<li>"
                "Orientation-specific meta-data in the image headers have "
                "been suspected as garbage and stripped-off to prevent severe "
                "mis-registration problems."
                "</li>")
        if do_bet:
            preproc_undergone += (
                "<li>"
                "Brain extraction has been applied to strip-off the skull"
                " and other non-brain tissues. This prevents later "
                "registration problems like the skull been (mis-)aligned "
                "unto the cortical surface, "
                "etc.</li>")
        if do_realign:
            preproc_undergone += (
                "<li>"
                "Motion correction has been done so as to correct for "
                "subject's head motion during the acquisition."
                "</li>")
        if do_coreg:
            preproc_undergone += (
                "<li>"
                "The subject's anatomical image has been coregistered "
                "against their fMRI images (precisely, to the mean thereof). "
                "Coregistration is important as it allows deformations of the "
                "anatomy to be directly applicable to the fMRI, or for ROIs "
                "to be defined on the anatomy."
                "</li>")
        if do_segment:
            preproc_undergone += (
                "<li>"
                "Tissue Segmentation has been employed to segment the "
                "anatomical image into GM, WM, and CSF compartments by using "
                "TPMs (Tissue Probability Maps) as priors.</li>")
        if do_normalize:
            preproc_undergone += (
                "<li>"
                "The segmented anatomical image has been warped "
                "into the MNI template space by applying the deformations "
                "learned during segmentation. The same deformations have been"
                " applied to the fMRI images.</li>")
        if fwhm:
            preproc_undergone += (
                "<li>"
                "The normalized fMRI images have been "
                "smoothed with a %smm x %smm x %smm "
                "Gaussian kernel.</li>") % tuple(fwhm)

        preproc_undergone += additional_preproc_undergone
        preproc_undergone += "</ul>"

        report_log_filename = os.path.join(
            subject_data.output_dir, 'report_log.html')
        report_preproc_filename = os.path.join(
            subject_data.output_dir, 'report_preproc.html')
        report_filename = os.path.join(
            subject_data.output_dir, 'report.html')

        final_thumbnail = reporter.Thumbnail()
        final_thumbnail.a = reporter.a(href=report_preproc_filename)
        final_thumbnail.img = reporter.img()
        final_thumbnail.description = subject_data.subject_id

        # initialize results gallery
        loader_filename = os.path.join(
            subject_data.output_dir, "results_loader.php")
        results_gallery = reporter.ResultsGallery(
            loader_filename=loader_filename,
            title="Report for subject %s" % subject_data.subject_id)
        output['results_gallery'] = results_gallery

        # initialize progress bar
        if subject_progress_logger is None:
            subject_progress_logger = reporter.ProgressReport(
                report_log_filename,
                other_watched_files=[report_filename,
                                     report_preproc_filename])
        output['progress_logger'] = subject_progress_logger

        # html markup
        log = reporter.FSL_SUBJECT_REPORT_LOG_HTML_TEMPLATE().substitute(
            results=results_gallery,
            start_time=time.ctime(),
            subject_id=subject_data.subject_id
            )
        preproc = reporter.FSL_SUBJECT_REPORT_PREPROC_HTML_TEMPLATE(
            ).substitute(
            results=results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            subject_id=subject_data.subject_id
            )
        main_html = reporter.FSL_SUBJECT_REPORT_HTML_TEMPLATE(
            ).substitute(
            results=results_gallery,
            start_time=time.ctime(),
            subject_id=subject_data.subject_id
            )

        with open(report_log_filename, 'w') as fd:
            fd.write(str(log))
            fd.close()
        with open(report_preproc_filename, 'w') as fd:
            fd.write(str(preproc))
            fd.close()
        with open(report_filename, 'w') as fd:
            fd.write(str(main_html))
            fd.close()

        def finalize_report():
            output['final_thumbnail'] = final_thumbnail

            if parent_results_gallery:
                commit_subject_thumnbail_to_parent_gallery(
                    final_thumbnail,
                    subject_data.subject_id,
                    parent_results_gallery)

            if last_stage:
                subject_progress_logger.finish(report_preproc_filename)
    else:
        results_gallery = None

    # brain extraction (bet)
    if do_bet:
        pass

    #####################
    #  motion correction
    #####################
    if do_realign:
        realign_output = _do_subject_realign(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            sessions=subject_data.session_id,
            do_report=do_report,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            in_files=subject_data.func,
            register_to_mean=True,
            jobtype='estwrite',
            ignore_exception=ignore_exception,
            )

        # collect output
        realign_result = realign_output['result']

        # if failed to realign, return
        if realign_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed realignment)'
                finalize_report()
            return

        subject_data.func = realign_result.outputs.realigned_files
        mean_func = realign_result.outputs.mean_image

        output['realign_result'] = realign_result
        output['estimated_motion'
               ] = realign_result.outputs.realignment_parameters

        # generate report stub
        if do_report:
            final_thumbnail.img.src = realign_output['rp_plot']
    else:
        # manually compute mean (along time axis) of fMRI images
        # XXX derive a more sensible path for the mean_func
        mean_func = os.path.join(subject_data.output_dir,
            'meanfunc')
        compute_mean_image(subject_data.func, output_filename=mean_func)

    ################################################################
    # co-registration of structural (anatomical) against functional
    ################################################################
    if do_coreg:
        # specify input files for coregistration
        comments = "anat -> epi"
        if func_to_anat:
            comments = 'epi -> anat'
            coreg_target = subject_data.anat
            coreg_source = mean_func
        else:
            coreg_target = mean_func
            coreg_jobtype = 'estimate'
            if subject_data.anat is None:
                if not subject_data.hires is None:
                    coreg_source = subject_data.hires
                else:
                    coreg_source = EPI_TEMPLATE
                coreg_jobtype = 'estwrite'
                do_segment = False
            else:
                coreg_source = subject_data.anat

        # run coreg proper
        coreg_output = _do_subject_coreg(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=do_report,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            comments=comments,
            target=coreg_target,
            source=coreg_source,
            jobtype=coreg_jobtype,
            ignore_exception=ignore_exception
            )

        # collect results
        coreg_result = coreg_output['result']
        output['coreg_result'] = coreg_result

        # if failed to coregister, return
        if coreg_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed coregistration)'
                finalize_report()
            return

        # rest anat to coregistered version thereof
        subject_data.anat = coreg_result.outputs.coregistered_source

        # generate report stub
        if do_report:
            final_thumbnail.img.src = coreg_output['axial_outline']

    ###################################
    # segmentation of anatomical image
    ###################################
    if do_segment:
        segment_data = subject_data.anat
        segment_output = _do_subject_segment(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=do_report,
            progress_logger=subject_progress_logger,
            data=segment_data,
            gm_output_type=[True, True, True],
            wm_output_type=[True, True, True],
            csf_output_type=[True, True, True],
            tissue_prob_maps=[GM_TEMPLATE,
                              WM_TEMPLATE, CSF_TEMPLATE],
            gaussians_per_class=[2, 2, 2, 4],
            affine_regularization="mni",
            bias_regularization=0.0001,
            bias_fwhm=60,
            warping_regularization=1,
            ignore_exception=ignore_exception
            )

        segment_result = segment_output['result']

        # if failed to segment, return
        if segment_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed segmentation)'
                finalize_report()
            return

        # output['segment_result'] = segment_result

        if do_normalize:
            ##############################################################
            # indirect normalization: warp fMRI images int into MNI space
            # using the deformations learned by segmentation
            ##############################################################
            norm_parameter_file = segment_result.outputs.transformation_mat
            norm_apply_to_files = subject_data.func

            norm_output = _do_subject_normalize(
                subject_data.output_dir,
                subject_id=subject_data.subject_id,
                segment_result=segment_result,
                do_report=do_report,
                results_gallery=results_gallery,
                progress_logger=subject_progress_logger,
                brain='epi',
                sessions=subject_data.session_id,
                fwhm=fwhm,
                parameter_file=norm_parameter_file,
                apply_to_files=norm_apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=get_vox_dims(norm_apply_to_files),
                write_interp=1,
                jobtype='write',
                ignore_exception=ignore_exception
                )

            norm_result = norm_output["result"]

            # if failed to normalize, return None
            if norm_result.outputs is None:
                if do_report:
                    final_thumbnail.img.src = 'failed.png'
                    final_thumbnail.description += ' (failed normalization)'
                    finalize_report()
                return

            subject_data.func = norm_result.outputs.normalized_files
            output['func'] = norm_output['normalized_files']

            if do_report:
                final_thumbnail.img.src = \
                    norm_output['axial']

            #########################################################
            # indirect normalization: warp anat image into MNI space
            # using the deformations learned by segmentation
            #########################################################
            import pylab as pl

            norm_parameter_file = segment_result.outputs.transformation_mat
            norm_apply_to_files = subject_data.anat

            norm_output = _do_subject_normalize(
                subject_data.output_dir,
                subject_id=subject_data.subject_id,
                segment_result=segment_result,
                brain="anat",
                cmap=pl.cm.gray,
                do_report=do_report,
                results_gallery=results_gallery,
                progress_logger=subject_progress_logger,
                parameter_file=norm_parameter_file,
                apply_to_files=norm_apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=get_vox_dims(norm_apply_to_files),
                write_wrap=[0, 0, 0],
                write_interp=1,
                jobtype='write',
                ignore_exception=ignore_exception
                )

            norm_result = norm_output['result']

            # if failed to normalize, return None
            if norm_result.outputs is None:
                if do_report:
                    final_thumbnail.img.src = 'failed.png'
                    final_thumbnail.description += ' (failed normalization)'
                    finalize_report()
                return

            output['anat'] = norm_result.outputs.normalized_files

    elif do_normalize:
        ############################################
        # learn T1 deformation without segmentation
        ############################################
        norm_output = _do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            source=subject_data.anat,
            template=T1_TEMPLATE,
            _report=False)

        norm_result = norm_output['result']

        # if failed to normalize, return None
        if norm_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                finalize_report()
            return

        ####################################################
        # Warp EPI into MNI space using learned deformation
        ####################################################
        norm_parameter_file = norm_result.outputs.normalization_parameters
        norm_apply_to_files = subject_data.func

        norm_output = _do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            brain="epi",
            fwhm=fwhm,
            do_report=do_report,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=get_vox_dims(norm_apply_to_files),
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write',
            ignore_exception=ignore_exception
            )

        norm_result = norm_output["result"]

        # if failed to normalize, return None
        if norm_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed normalization)'
                finalize_report()
            return

        subject_data.func = norm_result.outputs.normalized_files
        output['func'] = norm_result.outputs.normalized_files

        #####################################################
        # Warp anat into MNI space using learned deformation
        #####################################################
        import pylab as pl

        norm_apply_to_files = subject_data.anat

        norm_output = _do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            brain="anat",
            cmap=pl.cm.gray,
            do_report=do_report,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=get_vox_dims(norm_apply_to_files),
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write',
            ignore_exception=ignore_exception
            )

        norm_result = norm_output['result']

        # if failed to normalize, return None
        if norm_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed normalization)'
                finalize_report()
            return

        output['anat'] = norm_result.outputs.normalized_files

    if do_report:
        # generate cv plots
        if do_cv_tc and do_normalize:
            corrected_FMRI = output['func']

            generate_cv_tc_thumbnail(corrected_FMRI,
                                     subject_data.session_id,
                                     subject_data.subject_id,
                                     subject_data.output_dir,
                                     results_gallery=results_gallery)

        finalize_report()

    return subject_data, output


def _do_subject_dartelnorm2mni(output_dir,
                               structural_file,
                               functional_file,
                               native_gm_image=None,
                               sessions=None,
                               subject_id=None,
                               downsample_func=True,
                               do_report=True,
                               do_cv_tc=True,
                               final_thumbnail=None,
                               results_gallery=None,
                               parent_results_gallery=None,
                               subject_progress_logger=None,
                               last_stage=True,
                               ignore_exception=True,
                               **dartelnorm2mni_kwargs):
    """
    Uses spm.DARTELNorm2MNI to warp subject brain into MNI space.

    Parameters
    ----------
    output_dir: string
        existing directory; results will be cache here

    **dartelnorm2mni_kargs: parameter-value list
        options to be passes to spm.DARTELNorm2MNI back-end

    """

    output = {"subject_id": subject_id,
              "output_dir": output_dir}

    output['progress_logger'] = subject_progress_logger

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    dartelnorm2mni = mem.cache(spm.DARTELNorm2MNI)

    subject_gm_file = None

    if native_gm_image:
        # warp subject gm image (produced by Segment or NewSegment)
        # into MNI space
        dartelnorm2mni_result = dartelnorm2mni(apply_to_files=native_gm_image,
                                               **dartelnorm2mni_kwargs)
        subject_gm_file = dartelnorm2mni_result.outputs.normalized_files

    # warp functional image into MNI space
    # functional_file = do_3Dto4D_merge(functional_file)
    createwarped = mem.cache(spm.CreateWarped)
    createwarped_result = createwarped(
        image_files=functional_file,
        flowfield_files=dartelnorm2mni_kwargs['flowfield_files'],
        ignore_exception=ignore_exception
        )

    # if node failed, return None
    if createwarped_result.outputs is None:
        return

    warped_files = createwarped_result.outputs.warped_files

    # down-sample warped epi to save disk space ?
    if downsample_func:
        import numpy as np

        if type(warped_files) is str:
            warped_files = [warped_files]

        resampled_warped_files = []
        for warped_file in warped_files:
            warped_file = do_3Dto4D_merge(warped_file)

            # compute new vox dims to down-sample to
            new_vox_dims = (np.array(get_vox_dims(warped_file)) \
                + np.array(get_vox_dims(functional_file))) / 2.0

            # down-sample proper
            resampled_warped_file = resample_img(
                warped_file, new_vox_dims)
            resampled_warped_files.append(resampled_warped_file)

        warped_files = resampled_warped_files

    # do_QA
    if do_report and results_gallery:
        import pylab as pl

        generate_normalization_thumbnails(
            warped_files,
            output_dir,
            brain='epi',
            cmap=pl.cm.spectral,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger)

        # generate segmentation thumbs
        epi_thumbs = generate_segmentation_thumbnails(
                warped_files,
                output_dir,
                subject_gm_file=subject_gm_file,
                brain='epi',
                results_gallery=results_gallery,
                progress_logger=subject_progress_logger)

    # warp anat into MNI space
    dartelnorm2mni_result = dartelnorm2mni(
        apply_to_files=structural_file,
        ignore_exception=ignore_exception,
        **dartelnorm2mni_kwargs
        )

    # if node failed, return None
    if dartelnorm2mni_result.outputs is None:
        return

    # do_QA
    if do_report and results_gallery:
        import pylab as pl

        generate_normalization_thumbnails(
            dartelnorm2mni_result.outputs.normalized_files,
            output_dir,
            brain='anat',
            cmap=pl.cm.gray,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger)

        # generate segmentation thumbs
        generate_segmentation_thumbnails(
            dartelnorm2mni_result.outputs.normalized_files,
            output_dir,
            subject_gm_file=subject_gm_file,
            brain='anat',
            cmap=pl.cm.gray,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger)

        # finalize report
        if parent_results_gallery:
            final_thumbnail.img.src = epi_thumbs['axial']
            commit_subject_thumnbail_to_parent_gallery(
                final_thumbnail,
                subject_id,
                parent_results_gallery)


    if do_report:
        # generate cv plots
        if do_cv_tc:
            generate_cv_tc_thumbnail(createwarped_result.outputs.warped_files,
                                     sessions,
                                     subject_id,
                                     output_dir,
                                     results_gallery=results_gallery)
            
        # shutdown page reloader
        if last_stage:
            subject_progress_logger.finish()

    # collect results and return
    output['dartelnorm2mni_result'] = dartelnorm2mni_result
    output['createwarped_result'] = createwarped_result
    output['func'] = createwarped_result.outputs.warped_files
    output['anat'] = dartelnorm2mni_result.outputs.normalized_files

    output['results_gallery'] = results_gallery
    output['progress_logger'] = subject_progress_logger

    return output


def do_group_DARTEL(output_dir,
                    subject_ids,
                    session_ids,
                    structural_files,
                    functional_files,
                    subject_output_dirs=None,
                    do_report=False,
                    subject_final_thumbs=None,
                    subject_results_galleries=None,
                    subject_progress_loggers=None,
                    parent_results_gallery=None,
                    ignore_exception=True):
    """
    Undocumented API!

    """

    if subject_results_galleries is None:
        subject_results_galleries = [None] * len(structural_files)

    if subject_progress_loggers is None:
        subject_progress_loggers = [None] * len(structural_files)

    if subject_final_thumbs is None:
        subject_final_thumbs = [None] * len(structural_files)

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    # compute gm, wm, etc. structural segmentation using Newsegment
    newsegment = mem.cache(spm.NewSegment)
    tissue1 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 1),
               2, (True, True), (False, False))
    tissue2 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 2),
               2, (True, True), (False, False))
    tissue3 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 3),
               2, (True, False), (False, False))
    tissue4 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 4),
               3, (False, False), (False, False))
    tissue5 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 5),
               4, (False, False), (False, False))
    tissue6 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 6),
               2, (False, False), (False, False))
    newsegment_result = newsegment(
        channel_files=structural_files,
        tissues=[tissue1, tissue2, tissue3, tissue4, tissue5, tissue6],
        ignore_exception=ignore_exception
        )

    # if node failed, return None
    if newsegment_result.outputs is None:
        return

    # compute DARTEL template for group data
    dartel = mem.cache(spm.DARTEL)
    dartel_input_images = [tpms for tpms in \
                               newsegment_result.outputs.dartel_input_images
                           if tpms]
    dartel_result = dartel(
        image_files=dartel_input_images,)

    # warp individual brains into group (DARTEL) space
    native_gm_images = newsegment_result.outputs.native_class_images[0]
    results = joblib.Parallel(
        n_jobs=N_JOBS, verbose=100,
        pre_dispatch='1.5*n_jobs',  # for scalability over RAM
        )(joblib.delayed(
            _do_subject_dartelnorm2mni)(
                subject_output_dirs[j],
                structural_files[j],
                functional_files[j],
                native_gm_image=native_gm_images[j],
                sessions=session_ids[j],
                subject_id=subject_ids[j],
                do_report=do_report,
                final_thumbnail=subject_final_thumbs[j],
                results_gallery=subject_results_galleries[j],
                subject_progress_logger=subject_progress_loggers[j],
                parent_results_gallery=parent_results_gallery,
                ignore_exception=ignore_exception,
                modulate=False,  # don't modulate
                fwhm=0,  # don't smooth
                flowfield_files=dartel_result.outputs.dartel_flow_fields[j],
                template_file=dartel_result.outputs.final_template_file,
                )
          for j in xrange(
                len(subject_ids)))

    # do QA
    if do_report:
        pass

    return results


def do_subjects_preproc(subjects,
                        output_dir=None,
                        dataset_id="UNNAMED DATASET!",
                        do_deleteorient=False,
                        do_report=True,
                        do_export_report=False,
                        dataset_description=None,
                        report_filename=None,
                        do_shutdown_reloaders=False,
                        fwhm=0,
                        do_bet=False,
                        do_realign=True,
                        do_coreg=True,
                        do_segment=True,
                        do_normalize=True,
                        do_dartel=False,
                        do_cv_tc=True,
                        ignore_exception=True
                        ):

    """This functions doe intra-subject fMRI preprocessing on a
    group os subjects.

    Parameters
    ----------
    subjects: iterable of SubjectData objects

    report_filename: string (optional)
    if provided, an HTML report will be produced. This report is
    dynamic and its contents are updated automatically as more
    and more subjects are preprocessed.

    Returns
    -------
    list of Bunch dicts with keys: anat, func, and subject_id, etc.
    for each preprocessed subject

    """

    # sanitize input
    if do_dartel:
        do_segment = False
        do_normalize = False

    if do_report and report_filename is None:
        raise RuntimeError(
            ("You asked for reporting (do_report=True)  but specified"
             " an invalid report_filename (None)"))

    kwargs = {'do_deleteorient': do_deleteorient,
              'do_report': do_report,
              'do_realign': do_realign, 'do_coreg': do_coreg,
              'do_segment': do_segment, 'do_normalize': do_normalize,
              'do_cv_tc': do_cv_tc,
              'fwhm': fwhm,
              'ignore_exception': ignore_exception,
              'last_stage': not do_dartel,
              }

    if output_dir is None:
        if do_report:
            output_dir = os.path.dirname(report_filename)
        else:
            output_dir = os.path.abspath("runs_XYZ")

    # generate html report (for QA) as desired
    parent_results_gallery = None
    if do_report:
        import time
        import reporting.reporter as reporter

        # do some sanity
        shutil.copy(os.path.join(root_dir, 'reporting/css', 'fsl.css'),
                    os.path.dirname(report_filename))
        shutil.copy(os.path.join(root_dir, 'reporting/css', 'styles.css'),
                    os.path.dirname(report_filename))
        shutil.copy(os.path.join(root_dir, "reporting/images/failed.png"),
                                 os.path.dirname(report_filename))
        shutil.copy(os.path.join(root_dir, "reporting/images/logo.jpeg"),
                                 os.path.dirname(report_filename))

        preproc_undergone = """\
<p>All preprocessing has been done using <a href="%s">pypreprocess</a>,
 which is powered by <a href="%s">nipype</a>, and <a href="%s">SPM8</a>.
</p>""" % (PYPREPROCESS_URL, NIPYPE_URL, SPM8_URL)

        preproc_undergone += "<ul>"

        additional_preproc_undergone = ""
        if do_dartel:
            additional_preproc_undergone += (
                "<li>"
                "Group/Inter-subject Normalization has been done using the "
                "SPM8 <a href='%s'>DARTEL</a> to warp subject brains into "
                "MNI space. "
                "The idea is to register images by computing a &ldquo;flow"
                " field&rdquo; which can then be &ldquo;exponentiated"
                "&rdquo; to generate both forward and backward deformation"
                "s. Processing begins with the &ldquo;import&rdquo; "
                "step. This involves taking the parameter files "
                "produced by the segmentation (NewSegment), and writing "
                "out rigidly "
                "transformed versions of the tissue class images, "
                "such that they are in as close alignment as possible with"
                " the tissue probability maps. &nbsp; "
                "The next step is the registration itself. This involves "
                "the simultaneous registration of e.g. GM with GM, "
                "WM with WM and 1-(GM+WM) with 1-(GM+WM) (when needed, the"
                " 1- (GM+WM) class is generated implicitly, so there "
                "is no need to include this class yourself). This "
                "procedure begins by creating a mean of all the images, "
                "which is used as an initial template. Deformations "
                "from this template to each of the individual images "
                "are computed, and the template is then re-generated"
                " by applying the inverses of the deformations to "
                "the images and averaging. This procedure is repeated a "
                "number of times. &nbsp;Finally, warped "
                "versions of the images (or other images that are in "
                "alignment with them) can be generated. "
                "</li>") % DARTEL_URL

        preproc_undergone += additional_preproc_undergone + "</ul>"

        kwargs['additional_preproc_undergone'] = additional_preproc_undergone

        report_log_filename = os.path.join(
            output_dir, 'report_log.html')
        report_preproc_filename = os.path.join(
            output_dir, 'report_preproc.html')
        report_filename = os.path.join(
            output_dir, 'report.html')

        # initialize results gallery
        loader_filename = os.path.join(
            output_dir, "results_loader.php")
        parent_results_gallery = reporter.ResultsGallery(
            loader_filename=loader_filename,
            refresh_timeout=30000,   # 30 seconds
            )

        # initialize progress bar
        progress_logger = reporter.ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename,
                                 report_preproc_filename])

        # html markup
        log = reporter.FSL_DATASET_REPORT_LOG_HTML_TEMPLATE().substitute(
            start_time=time.ctime(),
            )
        preproc = reporter.FSL_DATASET_REPORT_PREPROC_HTML_TEMPLATE(
            ).substitute(
            results=parent_results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            )
        main_html = reporter.FSL_DATASET_REPORT_HTML_TEMPLATE(
            ).substitute(
            results=parent_results_gallery,
            start_time=time.ctime(),
            dataset_id=dataset_id,
            )

        with open(report_log_filename, 'w') as fd:
            fd.write(str(log))
            fd.close()
        with open(report_preproc_filename, 'w') as fd:
            fd.write(str(preproc))
            fd.close()
        with open(report_filename, 'w') as fd:
            fd.write(str(main_html))
            fd.close()

        def finalize_report():
            progress_logger.finish(report_preproc_filename)
            print "HTML report (dynamic) written to %s" % report_filename

        kwargs['main_page'] = "../../%s" % os.path.basename(report_filename)

        if not do_dartel:
            kwargs['parent_results_gallery'] = parent_results_gallery

    # preprocess the subjects proper
    results = joblib.Parallel(
        n_jobs=N_JOBS,
        pre_dispatch='1.5*n_jobs',  # for scalability over RAM
        verbose=100)(joblib.delayed(
            _do_subject_preproc)(
                subject_data, **kwargs) for subject_data in subjects)

    if do_dartel:
        # collect subject_ids and session_ids
        subject_ids = [output['subject_id'] for _, output in results]
        session_ids = [output['session_id'] for _, output in results]

        # collect structural files for DARTEL pipeline
        if do_coreg:
            structural_files = [
                output['coreg_result'].outputs.coregistered_source
                                for _, output in results]
        else:
            structural_files = [
                subject_data.anat for subject_data, _ in results]

        # collect functional files for DARTEL pipeline
        if do_realign:
            functional_files = [
                output['realign_result'].outputs.realigned_files
                                for _, output in results]
        else:
            functional_files = [subject_data.func for subject_data,
                                _ in results]

        # collect subject output dirs
        subject_output_dirs = [subject_data.output_dir
                               for subject_data, _ in results]

        # collect gallery related subject-specific stuff
        subject_final_thumbs = None
        subject_results_galleries = None
        subject_progress_loggers = None

        if do_report:
            subject_final_thumbs = [output['final_thumbnail']
                                    for _, output in results]
            subject_results_galleries = [output['results_gallery']
                                         for _, output in results]
            subject_progress_loggers = [output['progress_logger']
                                         for _, output in results]
            estimated_motion = dict((output["subject_id"], output['estimated_motion'])
                                     for _, output in results)

        # normalize brains to their own template space (DARTEL)
        results = do_group_DARTEL(
            output_dir,
            subject_ids,
            session_ids,
            structural_files,
            functional_files,
            subject_output_dirs,
            do_report=do_report,
            subject_final_thumbs=subject_final_thumbs,
            subject_results_galleries=subject_results_galleries,
            subject_progress_loggers=subject_progress_loggers,
            parent_results_gallery=parent_results_gallery,
            ignore_exception=ignore_exception,
            )

        # dump json results
        _results = []
        for item in results:
            subject_result = {}
            subject_result['subject_id'] = item['subject_id']
            subject_result['func'] = item['createwarped_result'
                                           ].outputs.warped_files
            subject_result['anat'] = item[
                'dartelnorm2mni_result'].outputs.normalized_files
            if do_realign:
                subject_result['estimated_motion'] = estimated_motion[item['subject_id']]
            subject_result['output_dir'] = item['output_dir']

            # dump result to json output file
            json_outfile = os.path.join(
                subject_result['output_dir'], 'infos.json')
            json.dump(subject_result, open(json_outfile, 'wb'))

            if do_report:
                subject_result['progress_logger'] = item['progress_logger']

                if do_shutdown_reloaders:
                    subject_result['progress_logger'].finish_all()

            _results.append(subject_result)

        results = _results
    else:
        # dump json results
        _results = []
        for x in results:
            if x is None:
                continue
            else:
                item = x[1]
            subject_result = {}
            subject_result['subject_id'] = item['subject_id']
            subject_result['func'] = item['func']
            subject_result['anat'] = item['anat']
            if do_realign:
                subject_result['estimated_motion'] = item['estimated_motion']
            subject_result['output_dir'] = item['output_dir']

            # dump result to json output file
            json_outfile = os.path.join(
                subject_result['output_dir'], 'infos.json')
            json.dump(subject_result, open(json_outfile, 'wb'))

            if do_report:
                subject_result['progress_logger'] = item['progress_logger']

                if do_shutdown_reloaders:
                    subject_result['progress_logger'].finish_all()

            _results.append(subject_result)

        results = _results

    if do_report:
        finalize_report()

        print "HTML report (dynamic) written to %s" % report_filename

    # export report (so it can be emailed and QA'ed offline, for example)
    if do_report:
        if do_export_report:
            if do_dartel:
                tag = "DARTEL_workflow"
            else:
                tag = "standard_workflow"
            reporter.export_report(os.path.dirname(report_filename),
                                   tag=tag)

    # return results (caller may need this)
    return results
