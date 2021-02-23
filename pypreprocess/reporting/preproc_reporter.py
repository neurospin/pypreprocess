"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

"""

import os
import glob
import re
import shutil
import subprocess
import time
import json
import matplotlib
import numpy as np
import pylab as pl
import nibabel
from distutils.version import LooseVersion
import joblib
from .check_preprocessing import (plot_registration,
                                  plot_segmentation,
                                  plot_spm_motion_parameters)
from ..time_diff import plot_tsdiffs, multi_session_time_slice_diffs
from ..io_utils import compute_mean_3D_image, is_niimg, sanitize_fwhm
from ..configure_spm import _configure_spm, _get_version_spm
from .base_reporter import (Thumbnail,
                            ResultsGallery,
                            ProgressReport,
                            a,
                            img,
                            lines2breaks,
                            get_dataset_report_html_template,
                            get_dataset_report_preproc_html_template,
                            get_subject_report_html_template,
                            get_subject_report_preproc_html_template,
                            PYPREPROCESS_URL, DARTEL_URL, ROOT_DIR,
                            commit_subject_thumnbail_to_parent_gallery,
                            get_dataset_report_log_html_template,
                            copy_web_conf_files)
import matplotlib.pyplot as plt


# misc
SPM_DIR = _configure_spm()
EPI_TEMPLATE = GM_TEMPLATE = T1_TEMPLATE = WM_TEMPLATE = CSF_TEMPLATE = None


def _set_templates(spm_dir=SPM_DIR):
    """
    Sets paths of templates (T1, GM, WM, etc.), so that post-segmenation,
    etc. reporting works well.

    """
    global EPI_TEMPLATE, T1_TEMPLATE, GM_TEMPLATE, WM_TEMPLATE, CSF_TEMPLATE

    spm_version = _get_version_spm(SPM_DIR)

    # Set the tpm and template paths according to SPM version
    if spm_version == 'spm12':
        template_path = 'toolbox/OldNorm'
        tpm_path = 'toolbox/OldSeg'
    else:
        template_path = 'templates'
        tpm_path = 'tpm'

    # configure template images
    EPI_TEMPLATE = os.path.join(SPM_DIR, template_path, 'EPI.nii')
    SPM_T1_TEMPLATE = os.path.join(SPM_DIR, template_path, 'T1.nii')
    T1_TEMPLATE = "/usr/share/data/fsl-mni152-templates/avg152T1.nii"
    if not os.path.isfile(T1_TEMPLATE):
        T1_TEMPLATE += '.gz'
        if not os.path.exists(T1_TEMPLATE):
            T1_TEMPLATE = SPM_T1_TEMPLATE
    GM_TEMPLATE = os.path.join(SPM_DIR, tpm_path, 'grey.nii')
    WM_TEMPLATE = os.path.join(SPM_DIR, tpm_path, 'white.nii')
    CSF_TEMPLATE = os.path.join(SPM_DIR, tpm_path, 'csf.nii')

# extention of web-related files (increment this as we support more
# and more file extensions for web business)
WEBBY_EXTENSION_PATTERN = ".*\.(?:png|jpeg|html|php|css|txt|rst|js|gif)$"


def get_nipype_report_filename(
    output_files_or_dir):
    if isinstance(output_files_or_dir, str):
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
    dcm2nii=False,
    deleteorient=False,
    fwhm=None, anat_fwhm=None,
    bet=False,
    slice_timing=False,
    realign=False,
    coregister=False,
    coreg_func_to_anat=False,
    segment=False,
    normalize=False,
    func_write_voxel_sizes=None,
    anat_write_voxel_sizes=None,
    dartel=False,
    additional_preproc_undergone="",
    command_line=None,
    details_filename=None,
    has_func=True,
    ):
    """
    Generates a brief description of the pipeline used in the preprocessing.

    Parameters
    ----------
    command_line: string, optional (None)
        exact command-line typed at the terminal to run the underlying
        preprocessing (useful if someone were to reproduce your results)

    """
    fwhm = sanitize_fwhm(fwhm)
    anat_fwhm = sanitize_fwhm(anat_fwhm)
    if dartel:
        normalize = False
        segment = False

    # which tools were used ?
    if tools_used is None:
        tools_used = (
            'All preprocessing was done using <a href="%s">pypreprocess</a>,'
            ' a collection of python scripts and modules for '
            'preprocessing functional and anatomical MRI data.' % (
                PYPREPROCESS_URL))
    preproc_undergone = "<p>%s</p>" % tools_used

    # what was actually typed at the command line ?
    if not command_line is None:
        preproc_undergone += "Command-line: <i>%s</i><br/>" % command_line
    preproc_undergone += (
        "<br>For each subject, the following preprocessing steps have "
        "been done:")

    preproc_undergone += "<ul>"
    if prepreproc_undergone:
        preproc_undergone += "<li>%s</li>" % prepreproc_undergone
    if dcm2nii:
        preproc_undergone += (
            "<li>"
            "dcm2nii has been used to convert input images from DICOM to nifti"
            " format"
            "</li>")
    if deleteorient:
        preproc_undergone += (
            "<li>"
            "Orientation-specific meta-data in the image headers have "
            "been suspected as garbage and stripped-off to prevent severe "
            "mis-registration problems."
            "</li>")
    if bet:
        preproc_undergone += (
            "<li>"
            "Brain extraction has been applied to strip-off the skull"
            " and other non-brain tissues. This prevents later "
            "registration problems like the skull been (mis-)aligned "
            "unto the cortical surface, "
            "etc.</li>")
    if slice_timing:
        preproc_undergone += (
            "<li>"
            "Slice-Timing Correction (STC) has been done to interpolate the "
            "BOLD signal in time, so that in the sequel we can safely pretend"
            " all 3D volumes within a TR (Repetition Time) were "
            "acquired simultaneously, an crucial assumption for any further "
            "analysis of the data (GLM, ICA, etc.). "
            "</li>"
            )
    if realign:
        preproc_undergone += (
            "<li>"
            "Motion correction has been done so as to estimate, and then "
            "correct for, subject's head motion."
            "</li>"
            )
    if coregister:
        preproc_undergone += "<li>"
        if coreg_func_to_anat:
            preproc_undergone += (
                "The subject's functional images have been coregistered "
                "to their anatomical image."
                )
        else:
            preproc_undergone += (
                "The subject's anatomical image has been coregistered "
                "against their functional images.")
        preproc_undergone += (
            " Coregistration is important as it allows: (1) segmentation of "
            "the functional via segmentation of the anatomical brain; "
            "(2) inter-subject registration via inter-anatomical registration,"
            " a trick referred to as 'Indirect Normalization'; "
            "(3) ROIs to be defined on the anatomy, making it "
            "possible for activation maps to be projected and appreciated"
            " thereupon."
            "</li>")
    if segment:
        preproc_undergone += (
            "<li>"
            "Tissue Segmentation has been employed to segment the "
            "anatomical image into GM, WM, and CSF compartments, using "
            "template TPMs (Tissue Probability Maps).</li>")
    if normalize:
        if segment:
            if has_func:
                salt = (" The same deformations have been "
                        'applied to the functional images.')
            else: salt = ""
            preproc_undergone += (
                "<li>"
                "The segmented anatomical image has been warped "
                "into the MNI template space by applying the deformations "
                "learnt during segmentation.%s</li>" % salt)
        else:
            if coregister:
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
    if dartel:
        preproc_undergone += (
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
    if normalize or dartel:
        if (not func_write_voxel_sizes is None or
            not anat_write_voxel_sizes is None):
            preproc_undergone += "<li>"
            sep = ""
            if not func_write_voxel_sizes is None:
                preproc_undergone += (
                    "Output functional images have been re-written with voxel "
                    "size %smm x %smm x %smm.") % tuple(
                    func_write_voxel_sizes)
                sep = " "
            if not anat_write_voxel_sizes is None:
                preproc_undergone += (
                    "%sThe output anatomical image has been re-written with "
                    "voxel "
                    "size %smm x %smm x %smm.") % tuple([sep] + list(
                    anat_write_voxel_sizes))
            preproc_undergone += "</li>"

    if additional_preproc_undergone:
        preproc_undergone += additional_preproc_undergone
    if np.sum(fwhm) > 0 and has_func:
        preproc_undergone += (
            "<li>"
            "The functional images have been "
            "smoothed with a %smm x %smm x %smm "
            "Gaussian kernel.</li>") % tuple(fwhm)
    if np.sum(anat_fwhm) > 0:
        preproc_undergone += (
            "<li>"
            "The anatomical image has been "
            "smoothed with a %smm x %smm x %smm "
            "Gaussian kernel.") % tuple(anat_fwhm)
        if segment:
            preproc_undergone += (
                " Warped TPMs have been smoothed with the same kernel.")
    if not details_filename is None:
        preproc_undergone += (
            " <a href=%s>See complete configuration used for preprocessing"
            " here</a>") % os.path.basename(details_filename)
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
        # print("Removing old %s." % dst)
        # shutil.rmtree(dst)
        pass

    # copy hierarchy
    print("Copying files directory structure from %s to %s" % (src, dst))
    shutil.copytree(src, dst, ignore=ignore_these)
    print("+++++++Done.")

    # zip the results (dst)
    if make_archive:
        dst_archive = dst + ".zip"
        print("Writing archive %s .." % dst_archive)
        print(subprocess.check_output(
            'cd %s; zip -r %s %s; cd -' % (os.path.dirname(dst),
                                           os.path.basename(dst_archive),
                                           os.path.basename(dst))))
        print("+++++++Done.")


def nipype2htmlreport(nipype_report_filename):
    """
    Converts a nipype.caching report (.rst) to html.

    """
    with open(nipype_report_filename, 'r') as fd:
        return lines2breaks(fd.readlines())


def get_nipype_report(nipype_report_filename):
    if isinstance(nipype_report_filename, str):
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
        target, source, procedure_name, output_dir, tooltip=None,
        execution_log_html_filename=None, results_gallery=None):
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
        thumb_desc += " (<a href=%s>see execution log</a>)" % (
            os.path.basename(execution_log_html_filename))

    # plot outline (edge map) of template on the
    # normalized image
    outline = os.path.join(output_dir,
                           "%s_on_%s_outline.png" % (target[1], source[1]))

    qa_mem.cache(plot_registration)(
        target[0], source[0], output_filename=outline,
        close=True, title="Outline of %s on %s" % (target[1], source[1]))

    # create thumbnail
    if results_gallery:
        thumbnail = Thumbnail(tooltip=tooltip)
        thumbnail.a = a(href=os.path.basename(outline))
        thumbnail.img = img(
            src=os.path.basename(outline), height="250px")
        thumbnail.description = thumb_desc

        results_gallery.commit_thumbnails(thumbnail)

    # plot outline (edge map) of the normalized image
    # on the SPM MNI template
    source, target = (target, source)
    outline = os.path.join(
        output_dir, "%s_on_%s_outline.png" % (target[1], source[1]))
    outline_axial = os.path.join(
        output_dir, "%s_on_%s_outline_axial.png" % (target[1], source[1]))
    qa_mem.cache(plot_registration)(
        target[0], source[0], output_filename=outline_axial, close=True,
        display_mode='z', title="Outline of %s on %s" % (target[1], source[1]))

    output['axial'] = outline_axial
    qa_mem.cache(plot_registration)(
        target[0], source[0], output_filename=outline, close=True,
        title="Outline of %s on %s" % (target[1], source[1]))

    # create thumbnail
    if results_gallery:
        thumbnail = Thumbnail(tooltip=tooltip)
        thumbnail.a = a(href=os.path.basename(outline))
        thumbnail.img = img(
            src=os.path.basename(outline), height="250px")
        thumbnail.description = thumb_desc
        results_gallery.commit_thumbnails(thumbnail)                           

    return output


def generate_normalization_thumbnails(
        normalized_files, output_dir, brain="EPI", tooltip=None,
        execution_log_html_filename=None, results_gallery=None):
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
    if isinstance(normalized_files, str):
        normalized = normalized_files
    else:
        mean_normalized_img = compute_mean_3D_image(normalized_files)
        normalized = mean_normalized_img
    return generate_registration_thumbnails(
        (T1_TEMPLATE, 'template'), (normalized, brain),
        "Normalization of %s" % brain, output_dir,
        execution_log_html_filename=execution_log_html_filename,
        results_gallery=results_gallery, tooltip=tooltip)


def generate_coregistration_thumbnails(
        target, source, output_dir, execution_log_html_filename=None,
        results_gallery=None, tooltip=None, comment=True):
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
    comments = " %s == > %s" % (source[1], target[1]) if comment else ""
    return generate_registration_thumbnails(
        target, source, "Coregistration %s" % comments,
        output_dir, execution_log_html_filename=execution_log_html_filename,
        results_gallery=results_gallery, tooltip=tooltip)


def generate_segmentation_thumbnails(
        normalized_files, output_dir, subject_gm_file=None,
        subject_wm_file=None, subject_csf_file=None, only_native=False,
        brain='func', comments="", execution_log_html_filename=None,
        cmap=None, tooltip=None, results_gallery=None):
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
    if isinstance(normalized_files, str):
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
    _brain = "%s_%s" % (comments, brain) if comments else brain

    # plot contours of template compartments on subject's brain
    if not only_native:
        template_compartments_contours = os.path.join(
            output_dir,
            "template_tpms_contours_on_%s.png" % _brain)
        template_compartments_contours_axial = os.path.join(
            output_dir,
            "template_compartments_contours_on_%s_axial.png" % _brain)
        qa_mem.cache(plot_segmentation)(
            normalized_file, GM_TEMPLATE, wm_filename=WM_TEMPLATE,
            csf_filename=CSF_TEMPLATE, display_mode='z', cmap=cmap,
            output_filename=template_compartments_contours_axial,
            title="template TPMs", close=True)
        qa_mem.cache(plot_segmentation)(
            normalized_file, gm_filename=GM_TEMPLATE,
            wm_filename=WM_TEMPLATE, csf_filename=CSF_TEMPLATE,
            output_filename=template_compartments_contours,
            cmap=cmap, close=True,
            title=("Template GM, WM, and CSF TPM contours on "
                   "subject's %s") % _brain)

        # create thumbnail
        if results_gallery:
            thumbnail = Thumbnail(tooltip=tooltip)
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
            "subject_tpms_contours_on_subject_%s.png" % _brain)
        subject_compartments_contours_axial = os.path.join(
            output_dir,
            "subject_tpms_contours_on_subject_%s_axial.png" % _brain)

        qa_mem.cache(plot_segmentation)(
            normalized_file, subject_gm_file, wm_filename=subject_wm_file,
            csf_filename=subject_csf_file, display_mode='z', cmap=cmap,
            output_filename=subject_compartments_contours_axial, close=True, title="subject TPMs")

        title_prefix = "Subject's GM"
        if subject_wm_file:
            title_prefix += ", WM"
        if subject_csf_file:
            title_prefix += ", and CSF"

        qa_mem.cache(plot_segmentation)(
            normalized_file, subject_gm_file, wm_filename=subject_wm_file,
            csf_filename=subject_csf_file, cmap=cmap, close=True,
            output_filename=subject_compartments_contours,
            title=("%s TPM contours on "
               "subject's %s") % (title_prefix, _brain))

        # create thumbnail
        if results_gallery:
            thumbnail = Thumbnail(tooltip=tooltip)
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


def generate_tsdiffana_thumbnail(image_files, sessions, subject_id,
                                 output_dir, results_gallery=None,
                                 tooltips=None):
    """Generate tsdiffana thumbnails

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
    # plot figures
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)
    results = qa_mem.cache(multi_session_time_slice_diffs)(image_files)
    axes = plot_tsdiffs(results, use_same_figure=False)
    figures = [ax.get_figure() for ax in axes]
    output_filename_template = os.path.join(
        output_dir, "tsdiffana_plot_{0}.png")
    output_filenames = [output_filename_template.format(i)
                        for i in range(len(figures))]

    for fig, output_filename in zip(figures, output_filenames):
            fig.savefig(output_filename, bbox_inches="tight", dpi=200)
            pl.close(fig)

    if tooltips is None:
        tooltips = [None] * len(output_filename)

    # create thumbnails
    thumbnails = []
    for output_filename, tooltip in zip(output_filenames, tooltips):
        thumbnail = Thumbnail(tooltip=tooltip)
        thumbnail.a = a(
            href=os.path.basename(output_filename))
        thumbnail.img = img(
            src=os.path.basename(output_filename), height="250px",
            width="600px")
        thumbnail.description = "tsdiffana ({0} sessions)".format(
            len(sessions))
        thumbnails.append(thumbnail)
    if results_gallery:
        results_gallery.commit_thumbnails(thumbnails)
    return thumbnails


def generate_realignment_thumbnails(
        estimated_motion, output_dir, sessions=None, tooltip=None,
        execution_log_html_filename=None, results_gallery=None):
    """Function generates thumbnails for realignment parameters."""
    sessions = [1] if sessions is None else sessions
    if isinstance(estimated_motion, str):
        estimated_motion = [estimated_motion]
    output = {}
    if isinstance(estimated_motion, str):
        estimated_motion = [estimated_motion]
    tmp = []
    for x in estimated_motion:
        if isinstance(x, str):
            x = np.loadtxt(x)
        tmp.append(x)
    lengths = [len(each) for each in tmp]
    estimated_motion = np.vstack(tmp)
    rp_plot = os.path.join(output_dir, 'rp_plot.png')
    plot_spm_motion_parameters(parameter_file=estimated_motion, 
                        output_filename=rp_plot, lengths=lengths, close=True, 
                        title="Plot of Estimated motion for %d sessions" % len(sessions))

    # create thumbnail
    if results_gallery:
        thumbnail = Thumbnail(tooltip=tooltip)
        thumbnail.a = a(href=os.path.basename(rp_plot))
        thumbnail.img = img(src=os.path.basename(rp_plot),
                            height="250px", width="600px")
        thumbnail.description = "Motion Correction"
        if not execution_log_html_filename is None:
            thumbnail.description += (" (<a href=%s>see execution "
            "log</a>)") % os.path.basename(
                execution_log_html_filename)
        results_gallery.commit_thumbnails(thumbnail)
        output['rp_plot'] = rp_plot

    return output


def generate_stc_thumbnails(original_bold, st_corrected_bold, output_dir,
                            voxel=None, sessions=None, results_gallery=None,
                            execution_log_html_filename=None, close=True):
    sessions = [1] if sessions is None else sessions
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def _sanitize_data(data):
        if isinstance(data, list):
            return np.array([_sanitize_data(x) for x in data])

        if is_niimg(data):
            data = np.rollaxis(data.get_data(), -1, start=0)
        elif isinstance(data, str):
            data = nibabel.load(data).get_data()
        return data

    def _get_vol_shape(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                return x.shape
            else:
                assert x.ndim > 3, x.ndim
                return _get_vol_shape(x[..., 0])
        elif isinstance(x, str):
            return _get_vol_shape(nibabel.load(x).get_data())
        elif is_niimg(x):
            return _get_vol_shape(x.get_data())
        else:
            return _get_vol_shape(x[0])

    def _get_time_series_from_voxel(x, voxel):
        assert len(voxel) == 3
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                return x[voxel[0], voxel[1], voxel[2]]
            else:
                assert x.ndim == 4 == len(voxel) + 1
                return x[voxel[0], voxel[1], voxel[2], :]
        elif is_niimg(x):
            return _get_time_series_from_voxel(x.get_data(), voxel)
        elif isinstance(x, str):
            return _get_time_series_from_voxel(nibabel.load(x), voxel)
        else:
            return np.array([_get_time_series_from_voxel(y, voxel) for y in x])

    if voxel is None:
        voxel = np.array(_get_vol_shape(original_bold)) // 2
    output = {}

    for session_id, o_bold, stc_bold in zip(sessions, original_bold,
                                            st_corrected_bold):

        stc_ts = _get_time_series_from_voxel(stc_bold, voxel)
        o_ts = _get_time_series_from_voxel(o_bold, voxel)
        output_filename = os.path.join(output_dir,
                                       'stc_plot_%s.png' % session_id)
        pl.figure()
        pl.plot(o_ts, 'o-')
        if LooseVersion(matplotlib.__version__) < LooseVersion("2.0"):
            pl.hold('on')
        pl.plot(stc_ts, 's-')
        pl.legend(('original BOLD', 'ST corrected BOLD'))
        pl.title("session %s: STC QA for voxel (%s, %s, %s)" % (
                session_id, voxel[0], voxel[1], voxel[2]))
        pl.xlabel('time (TR)')

        pl.savefig(output_filename, bbox_inches="tight", dpi=200)
        if close:
            pl.close()

        # create thumbnail
        if results_gallery:
            thumbnail = Thumbnail()
            thumbnail.a = a(href=os.path.basename(
                    output_filename))
            thumbnail.img = img(src=os.path.basename(
                    output_filename), height="250px", width="600px")
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
        sessions=None,
        tools_used=None,
        fwhm=None,
        did_bet=False,
        did_slice_timing=False,
        did_deleteorient=False,
        did_realign=True,
        did_coreg=True,
        func_to_anat=False,
        did_segment=True,
        did_normalize=True,
        tsdiffana=True,
        additional_preproc_undergone=None,
        parent_results_gallery=None,
        subject_progress_logger=None,
        conf_path='.',
        last_stage=True):
    sessions = ['UNKNOWN_SESSION'] if sessions is None else sessions
    output = {}
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
        deleteorient=did_deleteorient,
        fwhm=fwhm,
        bet=did_bet,
        slice_timing=did_slice_timing,
        realign=did_realign,
        coreg=did_coreg,
        coreg_func_to_anat=func_to_anat,
        segment=did_segment,
        normalize=did_normalize,
        additional_preproc_undergone=additional_preproc_undergone)

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
    for css_file in glob.glob(os.path.join(ROOT_DIR, "css/*.css")):
        shutil.copy(css_file, output_dir)
    for icon_file in glob.glob(os.path.join(ROOT_DIR, "icons/*.gif")):
        shutil.copy(icon_file, output_dir)
    for icon_file in glob.glob(os.path.join(ROOT_DIR, "images/*.png")):
        shutil.copy(icon_file, output_dir)
    for icon_file in glob.glob(os.path.join(ROOT_DIR, "images/*.jpeg")):
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
    if did_slice_timing and not original_bold is None and not \
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
            cmap=pl.cm.nipy_spectral,
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

    # generate tsdiffana plots
    if tsdiffana:
        generate_tsdiffana_thumbnail(
            func,
            sessions,
            subject_id,
            output_dir,
            results_gallery=results_gallery)

    # we're done
    finalize_report()

    return output


def make_nipype_execution_log_html(nipype_output_files,
                                   node_name, output_dir,
                                   brain_name="",
                                   progress_logger=None):
    brain_name = brain_name.replace(" ", "_")  # fix forissue 169
    execution_log = get_nipype_report(get_nipype_report_filename(
        nipype_output_files))
    execution_log_html_filename = os.path.join(
        output_dir,
        '%s%sexecution_log.html' % (
            node_name, "_%s_" % brain_name if brain_name else "")
        )

    open(execution_log_html_filename, 'w').write(
        execution_log)

    if progress_logger:
        progress_logger.log(
            '<b>%s</b><br/><br/>' % node_name)
        progress_logger.log(execution_log)
        progress_logger.log('<hr/>')

    return execution_log_html_filename
