"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

XXX TODO: Document this module.
"""

import sys
import os
import shutil
import commands
import re
import time
import joblib
import matplotlib as mpl
import pylab as pl
import nibabel
import numpy as np
from nipy.labs import viz
import nipy.labs.statistical_mapping as sm

import check_preprocessing
import io_utils

sys.path.append("..")
import external.tempita.tempita as tempita

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

# find package path
ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]

# extention of web-related files (increment this as we support more
# and more file extensions for web business)
WEBBY_EXTENSION_PATTERN = ".*\.(?:png|jpeg|html|php|css)$"

"""MISC"""
NIPY_URL = "http://nipy.sourceforge.net/nipy/stable/index.html"
SPM8_URL = "http://www.fil.ion.ucl.ac.uk/spm/software/spm8/"
PYPREPROCESS_URL = "https://github.com/neurospin/pypreprocess"
DARTEL_URL = ("http://www.fil.ion.ucl.ac.uk/spm/software/spm8/"
              "SPM8_Release_Notes.pdf")
NIPYPE_URL = "http://nipy.sourceforge.net/nipype/"


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
        print "Removing old %s." % dst
        shutil.rmtree(dst)

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


def GALLERY_HTML_MARKUP():
    """
    Function to generate markup for the contents of a <div id="results">
    type html element.

    """

    return tempita.HTMLTemplate("""\
{{for thumbnail in thumbnails}}
<div class="img">
  <a {{attr(**thumbnail.a)}}>
    <img {{attr(**thumbnail.img)}}/>
  </a>
  <div class="desc">{{thumbnail.description | html}}</div>
</div>
{{endfor}}""")


class a(tempita.bunch):
    """
    HTML anchor element.

    """

    pass


class img(tempita.bunch):
    """
    HTML image element.

    """

    pass


class Thumbnail(tempita.bunch):
    """
    Thumbnnail (HTML img + effects).

    """

    pass


class ResultsGallery(object):
    """
    Gallery of results (summarized by thumbnails).

    """

    def __init__(self, loader_filename,
                 refresh_timeout=10,  # seconds
                 title='Results',
                 description=None
                 ):
        self.loader_filename = loader_filename
        self.refresh_timeout = refresh_timeout
        self.title = title
        self.description = description

        # start with a clean slate
        if os.path.isfile(self.loader_filename):
            os.remove(self.loader_filename)

        # touch loader file
        fd = open(self.loader_filename, 'a')
        fd.close()

    def commit_results_from_filename(self, filename):
        with open(filename) as fd:
            divs = fd.read()
            fd.close()

            loader_fd = open(self.loader_filename, 'a')
            loader_fd.write(divs)
            loader_fd.close()

    def commit_thumbnails(self, thumbnails, id=None):
        if not type(thumbnails) is list:
            thumbnails = [thumbnails]

        self.raw = GALLERY_HTML_MARKUP().substitute(thumbnails=thumbnails)

        fd = open(self.loader_filename, 'a')
        fd.write(self.raw)
        fd.close()


def SUBJECT_PREPROC_REPORT_HTML_TEMPLATE():
    """
    Report template for subject preproc.

    """

    with open(os.path.join(ROOT_DIR, 'template_reports',
                           'subject_preproc_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def DATASET_PREPROC_REPORT_HTML_TEMPLATE():
    """
    Returns report template for dataset preproc.

    """
    with open(os.path.join(ROOT_DIR, 'template_reports',
                           'dataset_preproc_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_SUBJECT_REPORT_LOG_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_subject_report_log_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_SUBJECT_REPORT_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
                           'fsl_subject_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_SUBJECT_REPORT_PREPROC_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_subject_report_preproc_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_SUBJECT_REPORT_STATS_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_subject_report_stats_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_DATASET_REPORT_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
                           'fsl_dataset_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_DATASET_REPORT_PREPROC_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_dataset_report_preproc_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_DATASET_REPORT_STATS_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_dataset_report_stats_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_DATASET_REPORT_LOG_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_dataset_report_log_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def lines2breaks(lines):
    """
    Converts line breaks to HTML breaks.

    """

    if type(lines) is str:
        lines = lines.split('\n')

    log = "<br>".join(lines)

    return tempita.HTMLTemplate(log).content


class ProgressReport(object):

    def __init__(self, report_filename, other_watched_files=[]):
        self.report_filename = report_filename
        self.other_watched_files = other_watched_files

        open(self.report_filename, 'a').close()

    def log(self, msg):
        """Logs an html-formated stub to the report file

        Parameters
        ----------
        msg: string
            message to log

        """

        with open(self.report_filename, 'r') as i_fd:
            content = i_fd.read()
            i_fd.close()
            marker = '<!-- log_next_thing_here -->'
            content = content.replace(marker, msg + marker)
            with open(self.report_filename, 'w') as o_fd:
                o_fd.write(content)
                o_fd.close()

    def finish(self, report_filename=None):
        """Stops the automatic reloading (by the browser, etc.) of a given
         report page

         Parameters
         ----------
         report_filename: string (optinal)
             file URL of page to stop re-loading

        """

        if report_filename is None:
            report_filename = self.report_filename

        with open(report_filename, 'r') as i_fd:
            content = i_fd.read()
            i_fd.close()

            # prevent pages from reloaded automaticall henceforth
            meta_reloader = "<meta http\-equiv=refresh content=.+?>"
            content = re.sub(meta_reloader, "", content)

            old_state = ("<font color=red><i>STILL RUNNING .."
                         "</i><blink>.</blink></font>")
            new_state = "Ended: %s" % time.ctime()
            new_content = content.replace(old_state, new_state)
            with open(report_filename, 'w') as o_fd:
                o_fd.write(new_content)
                o_fd.close()

    def finish_all(self):
        """Stops the automatic re-loading of watched pages

        """

        self.finish()

        for filename in self.other_watched_files:
            self.finish(filename)

    def watch_file(self, filename):
        self.other_watched_files.append(filename)


def nipype2htmlreport(nipype_report_filename):
    """
    Converts a nipype.caching report (.rst) to html.

    """
    with open(nipype_report_filename, 'r') as fd:
        return lines2breaks(fd.readlines())


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

    result_gallery: ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

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

        mean_normalized_img = io_utils.compute_mean_3D_image(normalized_files)
        normalized = mean_normalized_img

    # nipype report
    nipype_html_report_filename = os.path.join(
        output_dir,
        '%s_normalize_nipype_report.html' % brain)

    if os.path.exists(nipype_report_filename):
        nipype_report = nipype2htmlreport(
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

    thumb_desc = "Normalization of %s" % _brain
    if os.path.exists(nipype_report_filename):
        thumb_desc += (" (<a href=%s>see execution"
                       " log</a>)") % (os.path.basename(
                nipype_html_report_filename))

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
        thumbnail = Thumbnail()
        thumbnail.a = a(href=os.path.basename(outline))
        thumbnail.img = img(
            src=os.path.basename(outline), height="250px")
        thumbnail.description = thumb_desc

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
        thumbnail = Thumbnail()
        thumbnail.a = a(href=os.path.basename(outline))
        thumbnail.img = img(
            src=os.path.basename(outline), height="250px")
        thumbnail.description = thumb_desc

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

    result_gallery: ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

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

        io_utils.compute_mean_3D_image(normalized_files,
                           output_filename=mean_normalized_file)
        normalized_file = mean_normalized_file

    # nipype report
    nipype_html_report_filename = os.path.join(
        output_dir,
        '%s_normalize_nipype_report.html' % brain)

    if os.path.exists(nipype_report_filename):
        nipype_report = nipype2htmlreport(
            nipype_report_filename)
        open(nipype_html_report_filename,
             'w').write(str(nipype_report))

    output = {}

    # prepare for smart caching
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    thumb_desc = "Segmentation of %s " % _brain
    if os.path.exists(nipype_report_filename):
        thumb_desc += (" (<a href=%s>see execution "
                       "log</a>)") % (os.path.basename(
                nipype_html_report_filename))

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
        thumbnail = Thumbnail()
        thumbnail.a = a(
            href=os.path.basename(template_compartments_contours))
        thumbnail.img = img(
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
            thumbnail = Thumbnail()
            thumbnail.a = a(
                href=os.path.basename(subject_compartments_contours))
            thumbnail.img = img(
                src=os.path.basename(subject_compartments_contours),
                height="250px")
            thumbnail.description = thumb_desc

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
    thumbnail: Thumbnail instance
        thumbnail to be committed

    subject_id: string
        subject_id for subject under inspection

    result_gallery: ResultsGallery instance (optional)
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

    result_gallery: ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    if type(image_files) is str:
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
        cv_tc_plot_outfile=cv_tc_plot_output_file,
        plot_diff=True)

    # create thumbnail
    thumbnail = Thumbnail()
    thumbnail.a = a(
        href=os.path.basename(cv_tc_plot_output_file))
    thumbnail.img = img(
        src=os.path.basename(cv_tc_plot_output_file), height="250px",
        width="600px")
    thumbnail.description = "Coefficient of Variation (%d sessions)"\
                                 % len(sessions)

    if results_gallery:
        results_gallery.commit_thumbnails(thumbnail)


def generate_realignment_thumbnails(
    rp_files,
    output_dir,
    sessions=[1],
    results_gallery=None,
    progress_logger=None):
    """

    """

    if type(rp_files) is str:
        rp_files = [rp_files]

    output = {}

    # nipype report
    nipype_report_filename = os.path.join(
        os.path.dirname(rp_files[0]),
        "_report/report.rst")
    nipype_html_report_filename = os.path.join(
        output_dir,
        'realign_nipype_report.html')

    if os.path.exists(nipype_report_filename):
        nipype_report = nipype2htmlreport(nipype_report_filename)
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
            thumbnail = Thumbnail()
            thumbnail.a = a(href=os.path.basename(rp_plot))
            thumbnail.img = img(src=os.path.basename(rp_plot),
                                         height="250px",
                                         width="600px")
            thumbnail.description = "Motion Correction"
            if os.path.exists(nipype_report_filename):
                thumbnail.description += (" (<a href=%s>see execution "
                "log</a>)") % os.path.basename(
                    nipype_html_report_filename)

            results_gallery.commit_thumbnails(thumbnail)

        output['rp_plot'] = rp_plot

    return output


def generate_level1_report(zmap, mask,
                           output_html_path,
                           title="level 1 stats",
                           threshold=0.001,
                           method='fpr', cluster_th=0, null_zmax='bonferroni',
                           null_smax=None, null_s=None, nmaxima=4,
                           cluster_pval=.05):
    """
    Parameters
    ----------
    zmap: image object
        z-map data image
    mask: image object
        brain mask defining ROI
    output_html_path, string,
                      path where the output html should be written
    threshold, float, optional
               (p-variate) frequentist threshold of the activation image
    method, string, optional
            to be chosen as height_control in
            nipy.labs.statistical_mapping
    cluster_th, scalar, optional,
             cluster size threshold
    null_zmax: optional,
               parameter for cluster level statistics (?)
    null_s: optional,
             parameter for cluster level statistics (?)
    nmaxima: optional,
             number of local maxima reported per supra-threshold cluster
    """

    # Compute cluster statistics
    nulls = {'zmax': null_zmax, 'smax': null_smax, 's': null_s}

    """
    if null_smax is not None:
        print "a"
        clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                          nulls=nulls)
        clusters = [c for c in clusters if c['cluster_pvalue']<cluster_pval]
    else:
        print "b"
        clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                          height_control=method.lower(),
                                          cluster_th=cluster_th, nulls=nulls)
    """
    clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                      nulls=nulls, cluster_th=cluster_th,)
    if clusters is not None:
        clusters = [c for c in clusters if c['cluster_pvalue'] < cluster_pval]

    #if clusters == None or info == None:
    #    print "No results were written for %s" % zmap_file_path
    #    return
    if clusters == None:
        clusters = []

    # Make HTML page
    output = open(output_html_path, mode="w")
    output.write("<center>\n")
    output.write("<b>%s</b>\n" % title)
    output.write("<table border = 1>\n")
    output.write("<tr><th colspan=4> Voxel significance </th>\
    <th colspan=3> Coordinates in MNI referential</th>\
    <th>Cluster Size</th></tr>\n")
    output.write("<tr><th>p FWE corr<br>(Bonferroni)</th>\
    <th>p FDR corr</th><th>Z</th><th>p uncorr</th>")
    output.write("<th> x (mm) </th><th> y (mm) </th><th> z (mm) </th>\
    <th>(voxels)</th></tr>\n")

    for cluster in clusters:
        maxima = cluster['maxima']
        size = cluster['size']
        for j in range(min(len(maxima), nmaxima)):
            temp = ["%.3f" % cluster['fwer_pvalue'][j]]
            temp.append("%.3f" % cluster['fdr_pvalue'][j])
            temp.append("%.2f" % cluster['zscore'][j])
            temp.append("%.3f" % cluster['pvalue'][j])
            for it in range(3):
                temp.append("%.0f" % maxima[j][it])
            if j == 0:
                # Main local maximum
                temp.append('%i' % size)
                output.write('<tr><th align="center">' + '</th>\
                <th align="center">'.join(temp) + '</th></tr>')
            else:
                # Secondary local maxima
                output.write('<tr><td align="center">' + '</td>\
                <td align="center">'.join(temp) + '</td><td></td></tr>\n')

    nclust = len(clusters)
    nvox = sum([clusters[k]['size'] for k in range(nclust)])

    output.write("</table>\n")
    output.write("Number of voxels: %i<br>\n" % nvox)
    output.write("Number of clusters: %i<br>\n" % nclust)

    if info is not None:
        output.write("Threshold Z = %.2f (%s control at %.3f)<br>\n" \
                     % (info['threshold_z'], method, threshold))
        output.write("Cluster size threshold p<%s" % cluster_pval)
    else:
        output.write("Cluster size threshold = %i voxels" % cluster_th)

    output.write("</center>\n")
    output.close()


def make_standalone_colorbar(vmin, vmax, colorbar_outfile=None):
    """Plots a stand-alone colorbar

    """

    fig = pl.figure(figsize=(6, 1))
    ax = fig.add_axes([0.05, 0.4, 0.9, 0.5])

    cmap = pl.cm.hot
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
    pl.savefig(colorbar_outfile)

    return cb


def generate_subject_stats_report(
    stats_report_filename,
    design_matrix,
    contrasts,
    z_maps,
    subject_id,
    mask,
    anat=None,
    anat_affine=None,
    threshold=2.3,
    cluster_th=0,
    start_time=None,
    progress_logger=None,
    ):
    """Generates a report summarizing the statistical methods and results

    Parameters
    ----------
    stats_report_filename: string:
        html file to which output (generated html) will be written

    design_matrix: 'nipy design matrix' object
        design matrix for experiment

    contrasts: dict
       dictionary of contrasts of interest; the keys are the contrast ids,
       the values are contrast values (lists)

    z_maps: dict
       dict with same keys as 'contrasts'; the values are paths of z-maps
       for the respective contrasts

    mask: 'nifti image object'
        brain mask for subject

    anat: 3D array (optional)
        brain image to serve bg unto which activation maps will be plotted;
        passed to viz.plot_map API

    anat_affine: 2D array (optional)
        affine data for the anat

    threshold: float (optional)
        threshold to be applied to activation maps voxel-wise

    cluster_th: int (optional)
        minimal voxel count for clusteres declared as 'activated'

    start_time: string (optiona)
        start time for the stats analysis (useful for the generated
        report page)

    progress_logger: ProgressLogger object (optional)
        handle for logging progress

    """

    # prepare for stats reporting
    output_dir = os.path.dirname(stats_report_filename)
    stats_report_filename = os.path.join(output_dir,
                                         "report_stats.html")

    if progress_logger:
        progress_logger.watch_file(stats_report_filename)

    design_thumbs = ResultsGallery(
        loader_filename=os.path.join(output_dir,
                                     "design.html")
        )
    activation_thumbs = ResultsGallery(
        loader_filename=os.path.join(output_dir,
                                     "activation.html")
        )

    methods = """
    GLM and inference have been done using <a href="%s">nipy</a>. Statistic \
    images have been thresholded at Z>%s voxel-level.
    """ % (NIPY_URL, threshold)

    if start_time is None:
        start_time = time.ctime()
    level1_html_markup = FSL_SUBJECT_REPORT_STATS_HTML_TEMPLATE(
        ).substitute(
        start_time=start_time,
        subject_id=subject_id,
        methods=methods)
    with open(stats_report_filename, 'w') as fd:
        fd.write(str(level1_html_markup))
        fd.close()

    if progress_logger:
        progress_logger.log("<b>Level 1 statistics</b><br/><br/>")

    # show design matrix
    ax = design_matrix.show()
    ax.set_position([.05, .25, .9, .65])
    dmat_outfile = os.path.join(output_dir, 'design_matrix.png')
    pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)
    thumb = Thumbnail()
    thumb.a = a(href=os.path.basename(dmat_outfile))
    thumb.img = img(src=os.path.basename(dmat_outfile),
                             height="400px",
                             )
    thumb.description = "Design Matrix"
    design_thumbs.commit_thumbnails(thumb)

    _vmax = 0
    _vmin = threshold
    for j in xrange(len(contrasts)):
        contrast_id = contrasts.keys()[j]
        contrast_val = contrasts[contrast_id]
        map_path = z_maps[contrast_id]
        z_map = nibabel.load(map_path)

        # get positive z_map
        pos_data = z_map.get_data() * (z_map.get_data() > 0)

        # compute cut_coords for viz.plot_map(..) API
        n_axials = 12
        delta_z_axis = 3
        z_axis_max = np.unravel_index(
            pos_data.argmax(), z_map.shape)[2]
        z_axis_min = np.unravel_index(
            pos_data.argmin(), z_map.shape)[2]
        z_axis_min, z_axis_max = (min(z_axis_min, z_axis_max),
                                  max(z_axis_max, z_axis_min))
        z_axis_min = min(z_axis_min, z_axis_max - delta_z_axis * n_axials)
        cut_coords = np.linspace(z_axis_min, z_axis_max, n_axials)

        # compute vmin and vmax
        vmax = pos_data.max()
        vmin = pos_data.min()

        # update colorbar endpoints
        _vmax = max(_vmax, vmax)

        # plot activation proper
        viz.plot_map(pos_data, z_map.get_affine(),
                     cmap=pl.cm.hot,
                     anat=anat,
                     anat_affine=anat_affine,
                     vmin=vmin,
                     vmax=vmax,
                     threshold=threshold,
                     slicer='z',
                     cut_coords=cut_coords,
                     black_bg=True,
                     )

        # store activation plot
        z_map_plot = os.path.join(output_dir,
                                  "%s_z_map.png" % contrast_id)
        pl.savefig(z_map_plot, dpi=200, bbox_inches='tight',
                   facecolor="k",
                   edgecolor="k")
        stats_table = os.path.join(output_dir,
                                   "%s_stats_table.html" % contrast_id)

        # create thumbnail for activation
        thumbnail = Thumbnail()
        thumbnail.a = a(href=os.path.basename(stats_table))
        thumbnail.img = img(
            src=os.path.basename(z_map_plot), height="250px",)
        thumbnail.description = "%s contrast: %s" % (contrast_id, contrast_val)
        activation_thumbs.commit_thumbnails(thumbnail)

        generate_level1_report(
            z_map, mask,
            stats_table,
            title=map_path,
            cluster_th=cluster_th,
            )

    # make colorbar for activations
    colorbar_outfile = os.path.join(output_dir,
                                    'activation_colorbar.png')
    make_standalone_colorbar(_vmin, _vmax, colorbar_outfile)

    # we're done, shut down all re-loaders
    if progress_logger:
        progress_logger.log('<hr/>')
        progress_logger.finish_all()

    # return generated html
    with open(stats_report_filename, 'r') as fd:
        stats_report = fd.read()
        fd.close()

        return stats_report


def generate_subject_preproc_report(
    func_files=None,
    anat_file=None,
    rp_files=None,
    output_dir='/tmp',
    subject_id="UNSPECIFIED!",
    sessions=['UNKNOWN_SESSION'],
    do_cv_tc=True,
    preproc_undergone="UNSPECIFIED!",
    subject_progress_logger=None,
    ):

    output = {}

    preproc_undergone = """\
<p>All preprocessing has been done using <a href="%s">pypreprocess</a>,
 which is powered by <a href="%s">nipype</a>, and <a href="%s">SPM8</a>.
</p>""" % (PYPREPROCESS_URL, NIPYPE_URL, SPM8_URL)

    report_log_filename = os.path.join(
        output_dir, 'report_log.html')
    report_preproc_filename = os.path.join(
        output_dir, 'report_preproc.html')
    report_filename = os.path.join(
        output_dir, 'report.html')

    shutil.copy(os.path.join(ROOT_DIR, 'css', 'fsl.css'),
                output_dir)
    shutil.copy(os.path.join(ROOT_DIR, "images/logo.jpeg"),
                output_dir)

    # initialize results gallery
    loader_filename = os.path.join(
        output_dir, "results_loader.php")
    results_gallery = ResultsGallery(
        loader_filename=loader_filename,
        title="Report for subject %s" % subject_id)
    output['results_gallery'] = results_gallery

    # initialize progress bar
    if subject_progress_logger is None:
        subject_progress_logger = ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename,
                                 report_preproc_filename])
    output['progress_logger'] = subject_progress_logger

    # html markup
    preproc = FSL_SUBJECT_REPORT_PREPROC_HTML_TEMPLATE(
        ).substitute(
        conf_path=".",
        results=results_gallery,
        start_time=time.ctime(),
        preproc_undergone=preproc_undergone,
        subject_id=subject_id,
        )
    main_html = FSL_SUBJECT_REPORT_HTML_TEMPLATE(
        ).substitute(
        conf_path=".",
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
    if rp_files:
        generate_realignment_thumbnails(
            rp_files,
            output_dir,
            sessions=sessions,
            results_gallery=results_gallery,
            )

    # generate epi normalizatipon thumbs
    generate_normalization_thumbnails(
        func_files,
        output_dir,
        brain="EPI",
        cmap=pl.cm.spectral,
        results_gallery=results_gallery)

    generate_segmentation_thumbnails(
        func_files,
        output_dir,
        cmap=pl.cm.spectral,
        brain="EPI",
        results_gallery=results_gallery,
        )

    # generate anat normalization thumbs
    if anat_file:
        generate_normalization_thumbnails(
            anat_file,
            output_dir,
            brain="anat",
            cmap=pl.cm.gray,
            results_gallery=results_gallery)

        generate_segmentation_thumbnails(
            anat_file,
            output_dir,
            cmap=pl.cm.gray,
            brain="anat",
            results_gallery=results_gallery,
            )

    # generate cv tc plots
    if do_cv_tc:
        generate_cv_tc_thumbnail(
            func_files,
            sessions,
            subject_id,
            output_dir,
            results_gallery=results_gallery)

    # we're done; shutdown all reloaders
    subject_progress_logger.finish_all()

    return output
