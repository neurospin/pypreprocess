"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

"""

import os
import glob
import re
import shutil
import numpy as np
import commands
import time
import joblib
import pylab as pl
import json
import nibabel
from .check_preprocessing import (plot_registration,
                                  plot_cv_tc,
                                  plot_segmentation,
                                  plot_spm_motion_parameters
                                  )
from ..io_utils import (compute_mean_3D_image,
                        is_3D,
                        is_niimg
                        )
from .base_reporter import (Thumbnail,
                            ResultsGallery,
                            a,
                            img,
                            lines2breaks,
                            get_dataset_report_html_template,
                            get_dataset_report_preproc_html_template,
                            get_subject_report_html_template,
                            get_subject_report_preproc_html_template,
                            PYPREPROCESS_URL,
                            ROOT_DIR
                            )

# set templates
SPM_DIR = '/i2bm/local/spm8'
if 'SPM_DIR' in os.environ:
    SPM_DIR = os.environ['SPM_DIR']
# assert os.path.exists(SPM_DIR), \
#     "nipype_preproc_smp_utils: SPM_DIR: %s,\
#  doesn't exist; you need to export SPM_DIR" % SPM_DIR
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
    prepreproc_undergone="",
    tools_used=None,
    do_deleteorient=False,
    fwhm=None,
    do_bet=False,
    do_slicetiming=False,
    do_realign=False,
    do_coreg=False,
    coreg_func_to_anat=False,
    do_segment=False,
    do_normalize=False,
    additional_preproc_undergone=""):
    """
    Generates a brief description of the pipeline used in the preprocessing.

    """

    if tools_used is None:
        tools_used = (
            'All preprocessing was done using <a href="%s">pypreprocess</a>,'
            ' a collection of python tools (scripts, modules, etc.) for '
            'preprocessing functional data.') % PYPREPROCESS_URL,

    preproc_undergone = "<p>%s</p>" % tools_used
    preproc_undergone += "<ul>"

    if prepreproc_undergone:
        preproc_undergone += "<li>%s</li>" % prepreproc_undergone

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
            "Slice-Timing Correction (STC) has been done to interpolate the "
            "BOLD signal in time, so that in the sequel we can safely pretend"
            " all 3D volumes within a TR (Repetition Time) were "
            "acquired simultaneously, an crucial assumption for any further "
            "analysis of the data (GLM, ICA, etc.). "
            "</li>"
            )
    if do_realign:
        preproc_undergone += (
            "<li>"
            "Motion correction has been done so as to estimate, and then "
            "correct for, subject's head motion during the acquisition."
            "</li>"
            )
    if do_coreg:
        preproc_undergone += "<li>"
        if coreg_func_to_anat:
            preproc_undergone += (
                "The subject's functional images have been coregistered "
                "to their anatomical image."
                )
        else:
            preproc_undergone += (
            "The subject's anatomical image has been coregistered "
            "against their functional images."
                )
        preproc_undergone += (
            " Coregistration is important as it allows: (1) segmentation of "
            "the functional via segmentation of the anatomical brain; "
            "(2) inter-subject registration via inter-anatomical registration,"
            " a trick referred to as 'Indirect Normalization'; "
            "(3) ROIs to be defined on the anatomy, making it "
            "possible for activation maps to be projected and appreciated"
            " thereupon."
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
                " applied to the functional images.</li>")
        else:
            if do_coreg:
                preproc_undergone += (
                    "<li>"
                    "Deformations from native to standard space have been "
                    "learnt on the anatomically brain. These deformations "
                    "have been used to warp the functional and anatomical "
                    "images into standard space.</li>")
            else:
                preproc_undergone += (
                "<li>"
                "The functional images have been warped from native to "
                "standard space via classical normalization.</li>")
    if additional_preproc_undergone:
        preproc_undergone += additional_preproc_undergone
    if not fwhm is None:
        if len(np.shape(fwhm)) == 0:
            fwhm = [fwhm] * 3

        if np.sum(fwhm) > 0:
            preproc_undergone += (
                "<li>"
                "The resulting functional images have been "
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
        return lines2breaks(fd.readlines())


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
    """
    Generates QA thumbnails post-registration.

    Parameters
    ----------
    target: tuple of length 2
        target[0]: string
            path to reference image used in the registration
        target[1]: string
            short name (e.g 'anat', 'epi', 'MNI', etc.) for the
            reference image
    source: tuple of length 2
        source[0]: string
            path to source image
        source[1]: string
            short name (e.g 'anat', 'epi', 'MNI', etc.) for the
            source image
    procedure_name: string
        name of, or short comments on, the registration procedure used
        (e.g 'anat ==> func', etc.)

    """

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

    qa_mem.cache(plot_registration)(
        target[0],
        source[0],
        output_filename=outline,
        title="Outline of %s on %s" % (
            target[1],
            source[1],
            ))

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
        "%s_on_%s_outline.png" % (target[1],
                                  source[1]))
    outline_axial = os.path.join(
        output_dir,
        "%s_on_%s_outline_axial.png" % (target[1],
                                        source[1]))

    qa_mem.cache(plot_registration)(
        target[0],
        source[0],
        output_filename=outline_axial,
        slicer='z',
        title="Outline of %s on %s" % (
            target[1],
            source[1]))

    output['axial'] = outline_axial

    qa_mem.cache(plot_registration)(
        target[0],
        source[0],
        output_filename=outline,
        title="Outline of %s on %s" % (
            target[1],
            source[1],
            ))

    # create thumbnail
    if results_gallery:
        thumbnail = Thumbnail()
        thumbnail.a = a(href=os.path.basename(outline))
        thumbnail.img = img(
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
        mean_normalized_img = compute_mean_3D_image(normalized_files)
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
    """
    Generates QA thumbnails post-coregistration.

    Parameters
    ----------
    target: tuple of length 2
        target[0]: string
            path to reference image used in theco registration
        target[1]: string
            short name (e.g 'anat', 'epi', 'MNI', etc.) for the
            reference image

    source: tuple of length 2
        source[0]: string
            path to source image
        source[1]: string
            short name (e.g 'anat', 'epi', 'MNI', etc.) for the
            source image

    """

    return generate_registration_thumbnails(
        target,
        source,
        "Coregistration %s => %s" % (source[1], target[1]),
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
    only_native=False,
    brain='func',
    comments="",
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

    result_gallery: ResultsGallery instance (optional)
        gallery to which thumbnails will be committed

    """

    if isinstance(normalized_files, basestring):
        normalized_file = normalized_files
    else:
        mean_normalized_file = os.path.join(output_dir,
                                            "%s.nii" % brain)

        compute_mean_3D_image(normalized_files,
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

    _brain = "(%s) %s" % (comments, brain) if comments else brain

    # plot contours of template compartments on subject's brain
    if not only_native:
        template_compartments_contours = os.path.join(
            output_dir,
            "template_tmps_contours_on_%s.png" % _brain)
        template_compartments_contours_axial = os.path.join(
            output_dir,
            "template_compartments_contours_on_%s_axial.png" % _brain)

        qa_mem.cache(plot_segmentation)(
            normalized_file,
            GM_TEMPLATE,
            wm_filename=WM_TEMPLATE,
            csf_filename=CSF_TEMPLATE,
            output_filename=template_compartments_contours_axial,
            slicer='z',
            cmap=cmap,
            title="template TPMs")

        qa_mem.cache(plot_segmentation)(
            normalized_file,
            gm_filename=GM_TEMPLATE,
            wm_filename=WM_TEMPLATE,
            csf_filename=CSF_TEMPLATE,
            output_filename=template_compartments_contours,
            cmap=cmap,
            title=("Template GM, WM, and CSF contours on "
                   "subject's %s") % _brain)

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

        output['axial'] = template_compartments_contours_axial

    # plot contours of subject's compartments on subject's brain
    if subject_gm_file:
        subject_compartments_contours = os.path.join(
            output_dir,
            "subject_tmps_contours_on_subject_%s.png" % _brain)
        subject_compartments_contours_axial = os.path.join(
            output_dir,
            "subject_tmps_contours_on_subject_%s_axial.png" % _brain)

        qa_mem.cache(plot_segmentation)(
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
        qa_mem.cache(plot_segmentation)(
            normalized_file,
            subject_gm_file,
            wm_filename=subject_wm_file,
            csf_filename=subject_csf_file,
            output_filename=subject_compartments_contours,
            cmap=cmap,
            title=("%s contours on "
               "subject's %s") % (title_prefix, _brain))

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

        if only_native:
            output['axial'] = subject_compartments_contours_axial

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

    if isinstance(image_files, basestring) or is_niimg(image_files):
        image_files = [image_files]
    else:
        if is_3D(image_files[0]):
            image_files = [image_files]

    assert len(sessions) == len(image_files)

    cv_tc_plot_output_file = os.path.join(
        output_dir,
        "cv_tc_plot.png")

    qa_mem.cache(
        plot_cv_tc)(
        image_files,
        sessions,
        subject_id,
        _output_dir=output_dir,
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
        plot_spm_motion_parameters(
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
            if not execution_log_html_filename is None:
                thumbnail.description += (" (<a href=%s>see execution "
                "log</a>)") % os.path.basename(
                    execution_log_html_filename)

            results_gallery.commit_thumbnails(thumbnail)

        output['rp_plot'] = rp_plot

    return output


def generate_stc_thumbnails(
    original_bold,
    st_corrected_bold,
    output_dir,
    voxel=None,
    sessions=[1],
    execution_log_html_filename=None,
    results_gallery=None,
    progress_logger=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def _sanitize_data(data):
        if isinstance(data, list):
            return np.array([_sanitize_data(x) for x in data])

        if is_niimg(data):
            data = np.rollaxis(data.get_data(), -1, start=0)
        elif isinstance(data, basestring):
            data = nibabel.load(data).get_data()

        # if data.ndim < 5:
        #     data = np.array([data])

        return data

    original_bold = np.rollaxis(_sanitize_data(original_bold), 1, 5)
    st_corrected_bold = np.rollaxis(_sanitize_data(st_corrected_bold), 1, 5)

    assert st_corrected_bold.shape == original_bold.shape, "%ss != %s" % (
        str(st_corrected_bold.shape), str(original_bold.shape))

    if voxel is None:
        voxel = np.array(original_bold.shape[1:-1]) // 2

    output = {}

    for session_id, o_bold, stc_bold in zip(sessions, original_bold,
                                            st_corrected_bold):

        output_filename = os.path.join(output_dir,
                                       'stc_plot_%s.png' % session_id)

        pl.figure()
        pl.plot(o_bold[voxel[0], voxel[1], voxel[2], ...], 'o-')
        pl.hold('on')
        pl.plot(stc_bold[voxel[0], voxel[1], voxel[2], ...], 's-')
        pl.legend(('original BOLD', 'ST corrected BOLD'))
        pl.title("session %s: STC QA for voxel (%s, %s, %s)" % (
                session_id, voxel[0], voxel[1], voxel[2]))
        pl.xlabel('time (TR)')

        pl.savefig(output_filename, bbox_inches="tight", dpi=200)

        # create thumbnail
        if results_gallery:
            thumbnail = Thumbnail()
            thumbnail.a = a(href=os.path.basename(
                    output_filename))
            thumbnail.img = img(src=os.path.basename(
                    output_filename),
                                              height="250px",
                                              width="600px")
            thumbnail.description = "Slice-Timing Correction"
            if not execution_log_html_filename is None:
                thumbnail.description += (" (<a href=%s>see execution "
                "log</a>)") % os.path.basename(
                    execution_log_html_filename)

            results_gallery.commit_thumbnails(thumbnail)

        output['stc_plot'] = output_filename

    return output


def generate_subject_preproc_report(
    func=None,
    anat=None,
    original_bold=None,
    st_corrected_bold=None,
    estimated_motion=None,
    gm=None,
    wm=None,
    csf=None,
    output_dir='/tmp',
    subject_id="UNSPECIFIED!",
    sessions=['UNKNOWN_SESSION'],
    tools_used=None,
    fwhm=None,
    did_bet=False,
    did_slicetiming=False,
    slice_order='ascending',
    interleaved=False,
    did_deleteorient=False,
    did_realign=True,
    did_coreg=True,
    func_to_anat=False,
    did_segment=True,
    did_normalize=True,
    do_cv_tc=True,
    additional_preproc_undergone=None,
    parent_results_gallery=None,
    subject_progress_logger=None,
    conf_path='.',
    last_stage=True,
    ):

    output = {}

    # sanity
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # generate explanation of preproc steps undergone by subject
    preproc_undergone = generate_preproc_undergone_docstring(
        tools_used=tools_used,
        do_deleteorient=did_deleteorient,
        fwhm=fwhm,
        do_bet=did_bet,
        do_slicetiming=did_slicetiming,
        do_realign=did_realign,
        do_coreg=did_coreg,
        coreg_func_to_anat=func_to_anat,
        do_segment=did_segment,
        do_normalize=did_normalize,
        additional_preproc_undergone=additional_preproc_undergone,
        )

    report_log_filename = os.path.join(
        output_dir, 'report_log.html')
    report_preproc_filename = os.path.join(
        output_dir, 'report_preproc.html')
    report_filename = os.path.join(
        output_dir, 'report.html')

    # copy css and js stuff to output dir
    for js_file in glob.glob(os.path.join(ROOT_DIR,
                                          "js/*.js")):
        shutil.copy(js_file, output_dir)
    for css_file in glob.glob(os.path.join(ROOT_DIR,
                                               "css/*.css")):
        shutil.copy(css_file, output_dir)
    for icon_file in glob.glob(os.path.join(ROOT_DIR,
                                            "icons/*.gif")):
        shutil.copy(icon_file, output_dir)
    for icon_file in glob.glob(os.path.join(ROOT_DIR,
                                            "images/*.png")):
        shutil.copy(icon_file, output_dir)
    for icon_file in glob.glob(os.path.join(ROOT_DIR,
                                            "images/*.jpeg")):
        shutil.copy(icon_file, output_dir)

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
    preproc = get_subject_report_preproc_html_template(
        ).substitute(
        conf_path=conf_path,
        results=results_gallery,
        start_time=time.ctime(),
        preproc_undergone=preproc_undergone,
        subject_id=subject_id,
        )
    main_html = get_subject_report_html_template(
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

    # generate stc thumbs
    if did_slicetiming and not original_bold is None and not \
            st_corrected_bold is None:
        generate_stc_thumbnails(original_bold,
                                st_corrected_bold,
                                output_dir,
                                sessions=sessions,
                                results_gallery=results_gallery
                                )

    # generate realignment thumbs
    if did_realign and not estimated_motion is None:
        generate_realignment_thumbnails(
            estimated_motion,
            output_dir,
            sessions=sessions,
            results_gallery=results_gallery,
            )

    # generate coreg thumbs
    if did_coreg:
        target, ref_brain = func, "func"
        source, source_brain = anat, "anat"
        generate_coregistration_thumbnails(
            (target, ref_brain),
            (source, source_brain),
            output_dir,
            results_gallery=results_gallery,
            )

    # generate epi normalization thumbs
    if did_normalize:
        generate_normalization_thumbnails(
            func,
            output_dir,
            brain="EPI",
            results_gallery=results_gallery)

        seg_thumbs = generate_segmentation_thumbnails(
            func,
            output_dir,
            subject_gm_file=gm,
            subject_wm_file=wm,
            subject_csf_file=csf,
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
                results_gallery=results_gallery)

            generate_segmentation_thumbnails(
                anat,
                output_dir,
                subject_gm_file=gm,
                subject_wm_file=wm,
                subject_csf_file=csf,
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
    subject_preproc_data: array-like of strings or dicts
        .json filenames containing dicts, or dicts (one per
        subject) keys should be 'subject_id', 'func', 'anat', and
        optionally: 'estimated_motion'
    output_dir: string
        directory to where all output will be written
    dataset_id: string, optional (default None)
        a short description of the dataset (e.g "ABIDE")
    replace_in_path: array_like of pairs, optional (default None)
        things to replace in all paths for example [('/vaprofic',
        '/mnt'), ('NYU', 'ABIDE')] will replace, in all paths,
        '/vaporific' with  and 'NYU' with 'ABIDE'. This is useful if the
        data was copied from one location to another (thus invalidating)
        all path references in the json files in subject_preproc_data,
        etc.
    preproc_undergone: string
        a string describing the preprocessing steps undergone. This
        maybe cleartext or basic html (the latter is adviced)

    """

    output = {}

    # sanitize output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # copy web stuff
    copy_web_conf_files(output_dir)

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
    log = get_dataset_report_log_html_template().substitute(
        start_time=time.ctime(),
        )
    preproc = get_dataset_report_preproc_html_template(
        ).substitute(
        results=parent_results_gallery,
        start_time=time.ctime(),
        preproc_undergone=preproc_undergone,
        )
    main_html = get_dataset_report_html_template(
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
        for j, s in zip(xrange(len(subject_preproc_data)),
                               subject_preproc_data):
            if isinstance(s, basestring):
                # read dict from json file
                json_data = json.load(open(s))
                s = dict((k, json_data[k])
                         for k in json_data.keys()
                         if k in ['func', 'anat',
                                  'estimated_motion',
                                  'gm', 'wm', 'csf',
                                  'subject_id',
                                  'preproc_undergone'])

                if replace_in_path:
                    # correct of file/directory paths
                    for k, v in s.iteritems():
                        for stuff in replace_in_path:
                            if len(stuff) == 2:
                                if isinstance(v, basestring):
                                    s[k] = v.replace(
                                        stuff[0], stuff[1])
                                else:
                                    # XXX I'm assuming list-like type
                                    s[k] = [x.replace(
                                            stuff[0], stuff[1])
                                            for x in v]
 
            if not 'subject_id' in s:
                s['subject_id'] = 'sub%5i' % j
            if not 'output_dir' in s:
                s['output_dir'] = os.path.join(output_dir, s['subject_id'])

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


def make_nipype_execution_log_html(nipype_output_files,
                                   node_name, output_dir,
                                   progress_logger=None):
    execution_log = get_nipype_report(get_nipype_report_filename(
            nipype_output_files))
    execution_log_html_filename = os.path.join(
        output_dir,
        '%s_execution_log.html' % node_name
        )

    open(execution_log_html_filename, 'w').write(
        execution_log)

    if progress_logger:
        progress_logger.log(
            '<b>%s</b><br/><br/>' % node_name)
        progress_logger.log(execution_log)
        progress_logger.log('<hr/>')

    return execution_log_html_filename
