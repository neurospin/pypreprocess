"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

"""

import sys
import os
import glob
import re
import shutil
import commands
import time
import joblib
import pylab as pl
import nibabel
import numpy as np
import json

import check_preprocessing
import io_utils
import base_reporter

# set templates
SPM_DIR = '/i2bm/local/spm8'
if 'SPM_DIR' in os.environ:
    SPM_DIR = os.environ['SPM_DIR']
assert os.path.exists(SPM_DIR), \
    "nipype_preproc_smp_utils: SPM_DIR: %s,\
 doesn't exist; you need to export SPM_DIR" % SPM_DIR
EPI_TEMPLATE = os.path.join(SPM_DIR, 'templates/EPI.nii')
T1_TEMPLATE = "/usr/share/data/fsl-mni152-templates/avg152T1.nii"
if not os.path.isfile(T1_TEMPLATE):
    T1_TEMPLATE += '.gz'
    if not os.path.exists(T1_TEMPLATE):
        T1_TEMPLATE = os.path.join(SPM_DIR, "templates/T1.nii")
GM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/grey.nii')
WM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/white.nii')
CSF_TEMPLATE = os.path.join(SPM_DIR, 'tpm/csf.nii')

# extention of web-related files (increment this as we support more
# and more file extensions for web business)
WEBBY_EXTENSION_PATTERN = ".*\.(?:png|jpeg|html|php|css|txt|rst|js|gif)$"


def get_nipype_report_filename(
    output_files_or_dir):
    if isinstance(output_files_or_dir, basestring):
        if os.path.isdir(output_files_or_dir):
            return os.path.join(output_files_or_dir,
                                "_report/report.rst")
        elif os.path.isfile(output_files_or_dir):
            return get_nipype_report_filename(
                os.path.dirname(output_files_or_dir))
        else:
            raise OSError(
                "%s is neither a file nor directory!" % output_files_or_dir)
    else:
        # assuming list-like type
        return get_nipype_report_filename(output_files_or_dir[0])


def generate_preproc_undergone_docstring(
    do_deleteorient=False,
    fwhm=None,
    do_bet=False,
    do_slicetiming=False,
    do_realign=True,
    do_coreg=True,
    coreg_func_to_anat=False,
    do_segment=True,
    do_normalize=True,
    additional_preproc_undergone=""):
    preproc_undergone = """\
    <p>All preprocessing has been done using <a href="%s">pypreprocess</a>,
    which is powered by <a href="%s">nipype</a>, and <a href="%s">SPM8</a>.
    </p>""" % (base_reporter.PYPREPROCESS_URL,
               base_reporter.NIPYPE_URL, base_reporter.SPM8_URL)

    preproc_undergone += "<ul>"
    if do_deleteorient:
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
    if do_slicetiming:
        preproc_undergone += (
            "<li>"
            "Slice-timing has been done to interpolate the BOLD signal in "
            "time,"
            " so that we can safely pretend all 3D volumes within a TR were "
            "acquired simultaneously, an assumption crusial to the GLM "
            " and inference stages."
            "</li>"
            )
    if do_realign:
        preproc_undergone += (
            "<li>"
            "Motion correction has been done so as to correct for "
            "subject's head motion during the acquisition."
            "</li>"
            )
    if do_coreg:
        preproc_undergone += "<li>"
        if not coreg_func_to_anat:
            preproc_undergone += (
                "The subject's anatomical image has been coregistered "
                "against their fMRI images (precisely, to the mean thereof). "
                )
        else:
            preproc_undergone += (
            "The subject's anatomical image has been coregistered "
            "against their fMRI images (precisely, to the mean thereof). "
                )
        preproc_undergone += (
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
        if do_segment:
            preproc_undergone += (
                "<li>"
                "The segmented anatomical image has been warped "
                "into the MNI template space by applying the deformations "
                "learnt during segmentation. The same deformations have been"
                " applied to the fMRI images.</li>")
        else:
            if do_coreg:
                preproc_undergone += (
                    "<li>"
                    "Deformations from native to standard space have been "
                    "learnt on the anatomically brain. These deformations "
                    "have been used to warp the fMRI into standard space."
                    "</li>")
            else:
                preproc_undergone += (
                "<li>"
                "The fMRI images have been warped from native to standard "
                "space via classical normalization.</li>")
    if additional_preproc_undergone:
        preproc_undergone += additional_preproc_undergone
    if fwhm:
        if max(list(fwhm)) > 0:
            preproc_undergone += (
                "<li>"
                "These images have been "
                "smoothed with a %smm x %smm x %smm "
                "Gaussian kernel.</li>") % tuple(fwhm)

    preproc_undergone += "</ul>"

    return preproc_undergone


def del_empty_dirs(s_dir):
    """
    Recursively deletes all empty subdirs fo given dir.

    Parameters
    ==========
    s_dir: string
    directory under inspection

    """
    b_empty = True
    for s_target in os.listdir(s_dir):
        s_path = os.path.join(s_dir, s_target)
        if os.path.isdir(s_path):
            if not del_empty_dirs(s_path):
                b_empty = False
        else:
            b_empty = False
        if b_empty:
            print('deleting: %s' % s_dir)
            shutil.rmtree(s_dir)

    return b_empty


def export_report(src, tag="", make_archive=True):
    """
    Exports a report (html, php, etc. files) , ignoring data
    files like *.nii, etc.

    Parameters
    ==========
    src: string
    directory contain report

    make_archive: bool (optional)
    should the final report dir (dst) be archived ?

    """

    def check_extension(f):
        return re.match(WEBBY_EXTENSION_PATTERN, f)

    def ignore_these(folder, files):
        return [f for f in files if \
                    (os.path.isfile(
                    os.path.join(folder, f)) and not check_extension(f))]

    # sanity
    dst = os.path.join(src, "frozen_report_%s" % tag)

    if os.path.exists(dst):
        # print "Removing old %s." % dst
        # shutil.rmtree(dst)
        pass

    # copy hierarchy
    print "Copying files directory structure from %s to %s" % (src, dst)
    shutil.copytree(src, dst, ignore=ignore_these)
    print "+++++++Done."

    # zip the results (dst)
    if make_archive:
        dst_archive = dst + ".zip"
        print "Writing archive %s .." % dst_archive
        print commands.getoutput(
            'cd %s; zip -r %s %s; cd -' % (os.path.dirname(dst),
                                           os.path.basename(dst_archive),
                                           os.path.basename(dst)))
        print "+++++++Done."


def nipype2htmlreport(nipype_report_filename):
    """
    Converts a nipype.caching report (.rst) to html.

    """
    with open(nipype_report_filename, 'r') as fd:
        return base_reporter.lines2breaks(fd.readlines())


def get_nipype_report(nipype_report_filename,
                        ):
    if isinstance(nipype_report_filename, basestring):
        if os.path.isfile(nipype_report_filename):
            nipype_report_filenames = [nipype_report_filename]
        else:
            nipype_report_filenames = []
    else:
        nipype_report_filenames = nipype_report_filename

    output = []
    for nipype_report_filename in nipype_report_filenames:
        if os.path.exists(nipype_report_filename):
            nipype_report = nipype2htmlreport(
                nipype_report_filename)
            output.append(nipype_report)

    output = "<hr/>".join(output)

    return output


def generate_registration_thumbnails(
    target,
    source,
    procedure_name,
    output_dir,
    execution_log_html_filename=None,
    results_gallery=None,
    ):

    output = {}

    # prepare for smart caching
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    thumb_desc = procedure_name
    if execution_log_html_filename:
        thumb_desc += (" (<a href=%s>see execution"
                       " log</a>)") % (os.path.basename(
                execution_log_html_filename))

    # plot outline (edge map) of template on the
    # normalized image
    outline = os.path.join(
        output_dir,
        "%s_on_%s_outline.png" % (target[1],
                                  source[1]))

    qa_mem.cache(check_preprocessing.plot_registration)(
        target[0],
        source[0],
        output_filename=outline,
        title="Outline of %s on %s" % (
            target[1],
            source[1],
            ))

    # create thumbnail
    if results_gallery:
        thumbnail = base_reporter.Thumbnail()
        thumbnail.a = base_reporter.a(href=os.path.basename(outline))
        thumbnail.img = base_reporter.img(
            src=os.path.basename(outline), height="250px")
        thumbnail.description = thumb_desc

        results_gallery.commit_thumbnails(thumbnail)

    # plot outline (edge map) of the normalized image
    # on the SPM MNI template
    source, target = (target, source)
    outline = os.path.join(
        output_dir,
        "%s_on_%s_outline.png" % (target[1],
                                  source[1]))
    outline_axial = os.path.join(
        output_dir,
        "%s_on_%s_outline_axial.png" % (target[1],
                                        source[1]))

    qa_mem.cache(check_preprocessing.plot_registration)(
        target[0],
        source[0],
        output_filename=outline_axial,
        slicer='z',
        title="Outline of %s on %s" % (
            target[1],
            source[1]))

    output['axial'] = outline_axial

    qa_mem.cache(check_preprocessing.plot_registration)(
        target[0],
        source[0],
        output_filename=outline,
        title="Outline of %s on %s" % (
            target[1],
            source[1],
            ))

    # create thumbnail
    if results_gallery:
        thumbnail = base_reporter.Thumbnail()
        thumbnail.a = base_reporter.a(href=os.path.basename(outline))
        thumbnail.img = base_reporter.img(
            src=os.path.basename(outline), height="250px")
        thumbnail.description = thumb_desc

        results_gallery.commit_thumbnails(thumbnail)

    return output


def generate_normalization_thumbnails(
    normalized_files,
    output_dir,
    brain="EPI",
    execution_log_html_filename=None,
    results_gallery=None,
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

    result_gallery: ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

    if isinstance(normalized_files, basestring):
        normalized = normalized_files
    else:
        mean_normalized_img = io_utils.compute_mean_3D_image(normalized_files)
        normalized = mean_normalized_img

    return generate_registration_thumbnails(
        (T1_TEMPLATE, 'template'),
        (normalized, brain),
        "Normalization of %s" % brain,
        output_dir,
        execution_log_html_filename=execution_log_html_filename,
        results_gallery=results_gallery,
        )


def generate_coregistration_thumbnails(target,
                                       source,
                                       output_dir,
                                       execution_log_html_filename=None,
                                       results_gallery=None,
                                       progress_logger=None,
                                       ):

    return generate_registration_thumbnails(
        target,
        source,
        "Coregistration %s -> %s" % (source[1], target[1]),
        output_dir,
        execution_log_html_filename=execution_log_html_filename,
        results_gallery=results_gallery,
        )


def generate_segmentation_thumbnails(
    normalized_files,
    output_dir,
    subject_gm_file=None,
    subject_wm_file=None,
    subject_csf_file=None,
    brain='EPI',
    execution_log_html_filename=None,
    cmap=None,
    results_gallery=None,
    ):
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

    result_gallery: base_reporter.ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

    if isinstance(normalized_files, basestring):
        normalized_file = normalized_files
    else:
        mean_normalized_file = os.path.join(output_dir,
                                            "%s.nii" % brain)

        io_utils.compute_mean_3D_image(normalized_files,
                           output_filename=mean_normalized_file)
        normalized_file = mean_normalized_file

    output = {}

    # prepare for smart caching
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    thumb_desc = "Segmentation of %s " % brain
    if execution_log_html_filename:
        thumb_desc += (" (<a href=%s>see execution "
                       "log</a>)") % (os.path.basename(
                execution_log_html_filename))

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
        thumbnail = base_reporter.Thumbnail()
        thumbnail.a = base_reporter.a(
            href=os.path.basename(template_compartments_contours))
        thumbnail.img = base_reporter.img(
            src=os.path.basename(template_compartments_contours),
            height="250px")
        thumbnail.description = thumb_desc

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
            thumbnail = base_reporter.Thumbnail()
            thumbnail.a = base_reporter.a(
                href=os.path.basename(subject_compartments_contours))
            thumbnail.img = base_reporter.img(
                src=os.path.basename(subject_compartments_contours),
                height="250px")
            thumbnail.description = thumb_desc

            results_gallery.commit_thumbnails(thumbnail)

    output['axials'] = {}
    output['axial'] = template_compartments_contours_axial

    return output


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

    result_gallery: base_reporter.ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    if isinstance(image_files, basestring):
        image_files = [image_files]
    else:
        if io_utils.is_3D(image_files[0]):
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
        _output_dir=output_dir,
        cv_tc_plot_outfile=cv_tc_plot_output_file,
        plot_diff=True)

    # create thumbnail
    thumbnail = base_reporter.Thumbnail()
    thumbnail.a = base_reporter.a(
        href=os.path.basename(cv_tc_plot_output_file))
    thumbnail.img = base_reporter.img(
        src=os.path.basename(cv_tc_plot_output_file), height="250px",
        width="600px")
    thumbnail.description = "Coefficient of Variation (%d sessions)"\
                                 % len(sessions)

    if results_gallery:
        results_gallery.commit_thumbnails(thumbnail)

    return thumbnail


def generate_realignment_thumbnails(
    estimated_motion,
    output_dir,
    sessions=[1],
    execution_log_html_filename=None,
    results_gallery=None,
    progress_logger=None):
    """Function generates thumbnails for realignment parameters
    (aka estimated motion)

    """

    if isinstance(estimated_motion, basestring):
        estimated_motion = [estimated_motion]

    output = {}

    for session_id, rp in zip(sessions, estimated_motion):
        rp_plot = os.path.join(
            output_dir, 'rp_plot_%s.png' % session_id)
        check_preprocessing.plot_spm_motion_parameters(
            rp,
            title="Plot of Estimated motion for session %s" % session_id,
            output_filename=rp_plot)

        # create thumbnail
        if results_gallery:
            thumbnail = base_reporter.Thumbnail()
            thumbnail.a = base_reporter.a(href=os.path.basename(rp_plot))
            thumbnail.img = base_reporter.img(src=os.path.basename(rp_plot),
                                         height="250px",
                                         width="600px")
            thumbnail.description = "Motion Correction"
            if not execution_log_html_filename is None:
                thumbnail.description += (" (<a href=%s>see execution "
                "log</a>)") % os.path.basename(
                    execution_log_html_filename)

            results_gallery.commit_thumbnails(thumbnail)

        output['rp_plot'] = rp_plot

    return output


def generate_subject_preproc_report(
    func=None,
    anat=None,
    estimated_motion=None,
    output_dir='/tmp',
    subject_id="UNSPECIFIED!",
    sessions=['UNKNOWN_SESSION'],
    do_cv_tc=True,
    preproc_undergone=None,
    conf_path='.',
    last_stage=True,
    parent_results_gallery=None,
    subject_progress_logger=None,
    ):

    output = {}

    # sanity
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def finalize_report():
        output['final_thumbnail'] = final_thumbnail

        if parent_results_gallery:
            base_reporter.commit_subject_thumnbail_to_parent_gallery(
                final_thumbnail,
                subject_id,
                parent_results_gallery)

        if last_stage:
            subject_progress_logger.finish(report_preproc_filename)
            subject_progress_logger.finish_all()

    if not preproc_undergone:
        preproc_undergone = """\
<p>All preprocessing has been done using <a href="%s">pypreprocess</a>,
 which is powered by <a href="%s">nipype</a>, and <a href="%s">SPM8</a>.
</p>""" % (base_reporter.PYPREPROCESS_URL,
           base_reporter.NIPYPE_URL, base_reporter.SPM8_URL)

    report_log_filename = os.path.join(
        output_dir, 'report_log.html')
    report_preproc_filename = os.path.join(
        output_dir, 'report_preproc.html')
    report_filename = os.path.join(
        output_dir, 'report.html')

    # copy css and js stuff to output dir
    for js_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                          "js/*.js")):
        shutil.copy(js_file, output_dir)
    for css_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                               "css/*.css")):
        shutil.copy(css_file, output_dir)
    for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                            "icons/*.gif")):
        shutil.copy(icon_file, output_dir)
    for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                            "images/*.png")):
        shutil.copy(icon_file, output_dir)
    for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                            "images/*.jpeg")):
        shutil.copy(icon_file, output_dir)

    # initialize results gallery
    loader_filename = os.path.join(
        output_dir, "results_loader.php")
    results_gallery = base_reporter.ResultsGallery(
        loader_filename=loader_filename,
        title="Report for subject %s" % subject_id)
    final_thumbnail = base_reporter.Thumbnail()
    final_thumbnail.a = base_reporter.a(href=report_preproc_filename)
    final_thumbnail.img = base_reporter.img()
    final_thumbnail.description = subject_id

    output['results_gallery'] = results_gallery

    # initialize progress bar
    if subject_progress_logger is None:
        subject_progress_logger = base_reporter.ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename,
                                 report_preproc_filename])
    output['progress_logger'] = subject_progress_logger

    # html markup
    preproc = base_reporter.get_subject_report_preproc_html_template(
        ).substitute(
        conf_path=conf_path,
        results=results_gallery,
        start_time=time.ctime(),
        preproc_undergone=preproc_undergone,
        subject_id=subject_id,
        )
    main_html = base_reporter.get_subject_report_html_template(
        ).substitute(
        conf_path=conf_path,
        start_time=time.ctime(),
        subject_id=subject_id
        )

    with open(report_preproc_filename, 'w') as fd:
        fd.write(str(preproc))
        fd.close()
    with open(report_filename, 'w') as fd:
        fd.write(str(main_html))
        fd.close()

    # generate realignment thumbs
    if estimated_motion:
        generate_realignment_thumbnails(
            estimated_motion,
            output_dir,
            sessions=sessions,
            results_gallery=results_gallery,
            )

    # generate epi normalizatipon thumbs
    generate_normalization_thumbnails(
        func,
        output_dir,
        brain="EPI",
        cmap=pl.cm.spectral,
        results_gallery=results_gallery)

    seg_thumbs = generate_segmentation_thumbnails(
        func,
        output_dir,
        cmap=pl.cm.spectral,
        brain="EPI",
        results_gallery=results_gallery,
        )
    final_thumbnail.img.src = seg_thumbs['axial']

    # generate anat normalization thumbs
    if anat:
        generate_normalization_thumbnails(
            anat,
            output_dir,
            brain="anat",
            cmap=pl.cm.gray,
            results_gallery=results_gallery)

        generate_segmentation_thumbnails(
            anat,
            output_dir,
            cmap=pl.cm.gray,
            brain="anat",
            results_gallery=results_gallery,
            )

    # generate cv tc plots
    if do_cv_tc:
        generate_cv_tc_thumbnail(
            func,
            sessions,
            subject_id,
            output_dir,
            results_gallery=results_gallery)

    # we're done
    finalize_report()

    return output


def generate_dataset_preproc_report(
    subject_preproc_data,
    output_dir="/tmp",
    dataset_id="UNSPECIFIED!",
    replace_in_path=None,
    n_jobs=None,
    preproc_undergone=None,
    last_stage=True,
    ):
    """Generates post-preproc reports for a dataset

    Parameters
    ----------
    subject_preproc_data: array-like of strings or dictsSubjectData objects
        .json filenames containing dicts, or dicts (one per
        subject) keys should be 'subject_id', 'func', 'anat', and
        optionally: 'estimated_motion'

    """

    output = {}

    # copy css and js stuff to output dir
    for js_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                          "js/*.js")):
        shutil.copy(js_file, output_dir)
    for css_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                               "css/*.css")):
        shutil.copy(css_file, output_dir)
    for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                            "icons/*.gif")):
        shutil.copy(icon_file, output_dir)
    for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                            "images/*.png")):
        shutil.copy(icon_file, output_dir)
    for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                            "images/*.jpeg")):
        shutil.copy(icon_file, output_dir)

    report_log_filename = os.path.join(
        output_dir, 'report_log.html')
    report_preproc_filename = os.path.join(
        output_dir, 'report_preproc.html')
    report_filename = os.path.join(
        output_dir, 'report.html')

        # initialize results gallery
    loader_filename = os.path.join(
        output_dir, "results_loader.php")
    parent_results_gallery = base_reporter.ResultsGallery(
        loader_filename=loader_filename,
        refresh_timeout=30,   # 30 seconds
        )

    # initialize progress bar
    progress_logger = base_reporter.ProgressReport(
        report_log_filename,
        other_watched_files=[report_filename,
                             report_preproc_filename])
    output['progress_logger'] = progress_logger

    # html markup
    log = base_reporter.get_dataset_report_log_html_template().substitute(
        start_time=time.ctime(),
        )
    preproc = base_reporter.get_dataset_report_preproc_html_template(
        ).substitute(
        results=parent_results_gallery,
        start_time=time.ctime(),
        preproc_undergone=preproc_undergone,
        )
    main_html = base_reporter.get_dataset_report_html_template(
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

    # factory to generate subjects
    def subject_factory():
        for s in subject_preproc_data:
            if isinstance(s, basestring):
                # read dict from json file
                s = dict((k, json.load(open(s))[k])
                         for k in ['func', 'anat',
                                   'estimated_motion',
                                   'output_dir',
                                   'subject_id',
                                   'preproc_undergone'])

                if replace_in_path:
                    # correct of file/directory paths
                    for k, v in s.iteritems():
                        for stuff in replace_in_path:
                            if isinstance(v, basestring):
                                s[k] = v.replace(
                                    stuff[0], stuff[1])
                            else:
                                # XXX I'm assuming list-like type
                                s[k] = [x.replace(
                                        stuff[0], stuff[1])
                                        for x in v]
 
            yield s

    # generate reports
    if n_jobs is None:
        n_jobs = len(subject_preproc_data)
        n_jobs = 1  # min(n_jobs, multiprocessing.cpu_count() / 4)

    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(
            generate_subject_preproc_report)(
            parent_results_gallery=parent_results_gallery,
            **s) for s in subject_factory())

    # done ?
    progress_logger.finish_all()

    # game over
    if last_stage:
        progress_logger.finish_all()

    return output
