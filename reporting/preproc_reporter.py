"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

"""

import sys
import os
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
from base_reporter import *

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
WEBBY_EXTENSION_PATTERN = ".*\.(?:png|jpeg|html|php|css|txt|rst)$"


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


def lines2breaks(lines):
    """
    Converts line breaks to HTML breaks.

    """

    if isinstance(lines, basestring):
        lines = lines.split('\n')

    log = "<br>".join(lines)

    return tempita.HTMLTemplate(log).content


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

    if isinstance(normalized_files, basestring):
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

    if isinstance(normalized_files, basestring):
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

    return thumbnail


def generate_realignment_thumbnails(
    estimated_motion,
    output_dir,
    sessions=[1],
    results_gallery=None,
    progress_logger=None):
    """

    """

    if isinstance(estimated_motion, basestring):
        estimated_motion = [estimated_motion]

    output = {}

    # nipype report
    nipype_report_filename = os.path.join(
        os.path.dirname(estimated_motion[0]),
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

    for session_id, rp in zip(sessions, estimated_motion):
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

    def finalize_report():
        output['final_thumbnail'] = final_thumbnail

        if parent_results_gallery:
            commit_subject_thumnbail_to_parent_gallery(
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
</p>""" % (PYPREPROCESS_URL,
           NIPYPE_URL, SPM8_URL)

    report_log_filename = os.path.join(
        output_dir, 'report_log.html')
    report_preproc_filename = os.path.join(
        output_dir, 'report_preproc.html')
    report_filename = os.path.join(
        output_dir, 'report.html')

    # copy css and js stuff to output dir
    shutil.copy(os.path.join(ROOT_DIR, 'css/fsl.css'),
                output_dir)
    shutil.copy(os.path.join(ROOT_DIR, 'js/jquery.min.js'),
                output_dir)
    shutil.copy(os.path.join(ROOT_DIR, "images/logo.jpeg"),
                output_dir)
    shutil.copy(os.path.join(ROOT_DIR, "images/failed.png"),
                output_dir)

    # initialize results gallery
    loader_filename = os.path.join(
        output_dir, "results_loader.php")
    results_gallery = ResultsGallery(
        loader_filename=loader_filename,
        title="Report for subject %s" % subject_id)
    final_thumbnail = Thumbnail()
    final_thumbnail.a = a(href=report_preproc_filename)
    final_thumbnail.img = img()
    final_thumbnail.description = subject_id

    output['results_gallery'] = results_gallery

    # initialize progress bar
    if subject_progress_logger is None:
        subject_progress_logger = ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename,
                                 report_preproc_filename])
    output['progress_logger'] = subject_progress_logger

    # html markup
    preproc = FLS_SUBJECT_REPORT_PREPROC_HTML_TEMPLATE(
        ).substitute(
        conf_path=conf_path,
        results=results_gallery,
        start_time=time.ctime(),
        preproc_undergone=preproc_undergone,
        subject_id=subject_id,
        )
    main_html = FLS_SUBJECT_REPORT_HTML_TEMPLATE(
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

    shutil.copy('reporting/css/fsl.css',
                output_dir)
    shutil.copy("reporting/images/logo.jpeg",
                output_dir)

    report_log_filename = os.path.join(
        output_dir, 'report_log.html')
    report_preproc_filename = os.path.join(
        output_dir, 'report_preproc.html')
    report_filename = os.path.join(
        output_dir, 'report.html')

        # initialize results gallery
    loader_filename = os.path.join(
        output_dir, "results_loader.php")
    parent_results_gallery = ResultsGallery(
        loader_filename=loader_filename,
        refresh_timeout=30,   # 30 seconds
        )

    # initialize progress bar
    progress_logger = ProgressReport(
        report_log_filename,
        other_watched_files=[report_filename,
                             report_preproc_filename])
    output['progress_logger'] = progress_logger

    # html markup
    log = FLS_DATASET_REPORT_LOG_HTML_TEMPLATE().substitute(
        start_time=time.ctime(),
        )
    preproc = FLS_DATASET_REPORT_PREPROC_HTML_TEMPLATE(
        ).substitute(
        results=parent_results_gallery,
        start_time=time.ctime(),
        preproc_undergone=preproc_undergone,
        )
    main_html = FLS_DATASET_REPORT_HTML_TEMPLATE(
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

    for s in subject_preproc_data:
        if isinstance(s, basestring):
            # read dict from json file
            s = dict((k, json.load(open(s))[k]) for k in ['func', 'anat',
                                                          'estimated_motion',
                                                          'output_dir',
                                                          'subject_id',
                                                          'preproc_undergone'])

        # generate subject report
        generate_subject_preproc_report(
            parent_results_gallery=parent_results_gallery,
            conf_path="..",
            **s
            )

    progress_logger.finish()

    if last_stage:
        progress_logger.finish_all()

    return output
