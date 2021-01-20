import string
import os
from nilearn.reporting.utils import figure_to_svg_quoted
from matplotlib.pyplot import cm
import joblib
from joblib import Memory
import numpy as np
from .check_preprocessing import *
from ..time_diff import plot_tsdiffs, multi_session_time_slice_diffs
from ..io_utils import compute_mean_3D_image, sanitize_fwhm
from ..configure_spm import _configure_spm, _get_version_spm
from nilearn.plotting.html_document import HTMLDocument


HTML_TEMPLATE_ROOT_PATH = os.path.join(os.path.dirname(__file__),
                                           'template_reports')
SPM_DIR = _configure_spm()
# EPI_TEMPLATE = GM_TEMPLATE = T1_TEMPLATE = WM_TEMPLATE = CSF_TEMPLATE = None

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

def embed_in_HTML(html_template_file,components_to_embed):

    """ Embeds components in a given HTML template """

    html_template_path = os.path.join(HTML_TEMPLATE_ROOT_PATH,
                                      html_template_file)
    with open(html_template_path) as html_file_obj:
        html_template_text = html_file_obj.read()
    string_template = string.Template(html_template_text)

    string_text = string_template.safe_substitute(**components_to_embed)

    return string_text


def create_report(nilearn_report, output_dir, filename='nilearn_report.html'):

    html_outfile = os.path.join(output_dir, 'nilearn_report.html')

    nilearn_report['all_components'] = '\n'.join(
                                    nilearn_report['all_components'])
    nilearn_report_text = embed_in_HTML(
                                'nilearn_report_template.html', nilearn_report)
    nilearn_report_HTML = HTMLDocument(nilearn_report_text)

    nilearn_report_HTML.save_as_html(html_outfile)

    return 'Nilearn-style report created: {}'.format(html_outfile)


def _plot_to_svg(plot):
    """
    Creates an SVG image as a data URL
    from a Matplotlib Axes or Figure object.

    Parameters
    ----------
    plot: Matplotlib Axes or Figure object
        Contains the plot information.

    Returns
    -------
    url_plot_svg: String
        SVG Image Data URL
    """
    try:
        return figure_to_svg_quoted(plot)
    except AttributeError:
        return figure_to_svg_quoted(plot.figure)
        

def generate_realignment_report(subject_data,estimated_motion, output_dir
    , tooltip=None, execution_log_html_filename=None
    , nilearn_report=None):

    """ Creates plots associated with realignment 
    and returns it as an SVG url. """

    subject_data._set_session_ids()
    if not hasattr(subject_data, 'realignment_parameters'):
        raise ValueError("'realignment_parameters' attribute not set!")

    for_substitution = {}

    sessions = [1] if subject_data.session_ids is None else subject_data.session_ids
    if isinstance(estimated_motion, str):
        estimated_motion = [estimated_motion]
    tmp = []
    for x in estimated_motion:
        if isinstance(x, str):
            x = np.loadtxt(x)
        tmp.append(x)
    lengths = [len(each) for each in tmp]
    estimated_motion = np.vstack(tmp)

    for_substitution['plot'] = plot_spm_motion_parameters(
        parameter_file=estimated_motion
        , lengths=lengths, close=True, nilearn_report=nilearn_report,
        title="Plot of Estimated motion for %d sessions" % len(sessions))
    for_substitution['heading'] = "Plot of Estimated motion for %d sessions" % len(sessions)

    rp_plot_text = embed_in_HTML('report_sub_template.html'
                                ,for_substitution)

    return rp_plot_text

def generate_registration_report(target, source, procedure_name,
    output_dir, execution_log_html_filename=None, nilearn_report=None):

    """ Plots target's outline on source image and returns them 
    as SVG url embedded in HTML. """

    reg_plot = []
    for_substitution = {}

    # prepare for smart caching
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    # plot outline (edge map) of template on the
    # normalized image
    for_substitution['heading'] = procedure_name
    for_substitution['plot'] = qa_mem.cache(plot_registration)(
        target[0], source[0]
        , close=True, nilearn_report=nilearn_report,
        title="Outline of %s on %s" % (target[1], source[1]))
    reg_plot_text = embed_in_HTML('report_sub_template.html'
                                    ,for_substitution)
    reg_plot.append(reg_plot_text)

    # plot outline (edge map) of the normalized image
    # on the SPM MNI template
    source, target = (target, source)
    for_substitution['plot'] = qa_mem.cache(plot_registration)(
        target[0], source[0], close=True, nilearn_report=nilearn_report,
        title="Outline of %s on %s" % (target[1], source[1]))
    for_substitution['heading'] = ""
    reg_plot_text = embed_in_HTML('report_sub_template.html'
                                    ,for_substitution)
    reg_plot.append(reg_plot_text)

    return '\n'.join(reg_plot)


def generate_corregistration_report(subject_data, output_dir
    , coreg_func_to_anat=True, execution_log_html_filename=None
    , tooltip=None, comment=True, nilearn_report=None):

    """ Creates plots associated with corregistration 
    and returns them as SVG url embedded in HTML.
    Calls generate_registration_plot. """

    subject_data._set_session_ids()

    if subject_data.anat is None:
        print("Subject 'anat' field is None; nothing to do")
        return

    src, ref = subject_data.func, subject_data.anat
    src_brain, ref_brain = "mean_functional_image", "anatomical_image"
    if not coreg_func_to_anat:
        src, ref = ref, src
        src_brain, ref_brain = ref_brain, src_brain

    comments = " %s == > %s" % (src_brain, ref_brain)
    return generate_registration_report((ref, ref_brain)
        , (src, src_brain), "Coregistration %s" % comments,
        output_dir, nilearn_report=nilearn_report)


def generate_segmentation_report(subject_data, output_dir
    , subject_gm_file=None, subject_wm_file=None, subject_csf_file=None
    , comment="", only_native=False, tooltip=None
    , execution_log_html_filename=None, nilearn_report=None):
    
    """ Creates plots associated with segmentation 
    and returns them as SVG url embedded in HTML. """
    _set_templates()

    seg_plot = []
    for_substitution = {}

    subject_data._set_session_ids()

    segmented = False
    for item in ['gm', 'wm', 'csf']:
        if hasattr(subject_data, item):
            segmented = True
            break
    if not segmented:
        return

    for brain_name, brain, cmap in zip(
            ['anatomical_image', 'mean_functional_image'],
            [subject_data.anat, subject_data.func],
            [cm.gray, cm.nipy_spectral]):
        if not brain:
            continue

        if isinstance(brain, str):
            brain = brain
        else:
            mean_normalized_file = os.path.join(output_dir,
                                                "%s.nii" % brain_name)
            compute_mean_3D_image(brain,
                               output_filename=mean_normalized_file)
            brain = mean_normalized_file

        # prepare for smart caching
        qa_cache_dir = os.path.join(output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

        _brain_name = "%s_%s" % (comment, brain_name)
        for_substitution['heading'] = "Segmentation of %s " % _brain_name

        # plot contours of template compartments on subject's brain
        if not only_native:
            for_substitution['plot']=qa_mem.cache(plot_segmentation)(
                brain, gm_filename=GM_TEMPLATE,
                wm_filename=WM_TEMPLATE, csf_filename=CSF_TEMPLATE,
                cmap=cmap, close=True, nilearn_report=nilearn_report,
                title=("Template GM, WM, and CSF TPM contours on "
                    "subject's %s") % _brain_name)

            seg_plot_text = embed_in_HTML('report_sub_template.html'
                                    ,for_substitution)
            seg_plot.append(seg_plot_text)

        # plot contours of subject's compartments on subject's brain
        if subject_gm_file:
            title_prefix = "Subject's GM"
            if subject_wm_file:
                title_prefix += ", WM"
            if subject_csf_file:
                title_prefix += ", and CSF"

            for_substitution['plot'] = qa_mem.cache(plot_segmentation)(
                brain, subject_gm_file, wm_filename=subject_wm_file,
                csf_filename=subject_csf_file, cmap=cmap, close=True,
                nilearn_report=nilearn_report, title=("%s TPM contours on "
                    "subject's %s") % (title_prefix, _brain_name))
            if not only_native:
                for_substitution['heading'] = ""

            seg_plot_text = embed_in_HTML('report_sub_template.html'
                                    ,for_substitution)
            seg_plot.append(seg_plot_text)

    return '\n'.join(seg_plot)

def generate_normalization_report(subject_data, output_dir
    , tooltip=None, execution_log_html_filename=None
    , nilearn_report=None):
    
    """ Creates plots associated with normalization 
    and returns them as SVG url embedded in HTML. 
    Calls generate_segmentation_report 
    and generate_registration_report. """

    norm_plot = []
    subject_data._set_session_ids()

    warped_tpms = dict(
        (tpm, getattr(subject_data, tpm, None))
        for tpm in ["mwgm", "mwwm", "mwcsf"])
    segmented = warped_tpms.values().count(None) < len(warped_tpms)

    if segmented:
        norm_plot.append(generate_segmentation_report(
                subject_data=subject_data,
                output_dir=subject_data.output_dir,
                subject_gm_file=warped_tpms["mwgm"],
                subject_wm_file=warped_tpms["mwwm"],
                subject_csf_file=warped_tpms["mwcsf"],
                comment="warped",nilearn_report=nilearn_report))

    for brain_name, brain, cmap in zip(
            ['anatomical_image', 'mean_functional_image'],
            [subject_data.anat, subject_data.func],
            [cm.gray, cm.nipy_spectral]):
        if not brain:
            continue

        if isinstance(brain, str):
            normalized = brain
        else:
            mean_normalized_img = compute_mean_3D_image(brain)
            normalized = mean_normalized_img

        norm_plot.append(generate_registration_report(
        (T1_TEMPLATE, 'template'), (normalized, brain_name),
        "Normalization of %s" % brain_name, output_dir,
        nilearn_report=nilearn_report))

    return '\n'.join(norm_plot)

def generate_tsdiffana_report(image_files, sessions, subject_id,
                            output_dir, tooltips=None):

    """ Creates plots associated with tsdiffana 
    and returns them as SVG url embedded in HTML. """

    tsdiffana_plot = []
    for_substitution = {}
    
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)
    results = qa_mem.cache(multi_session_time_slice_diffs)(image_files)
    # plot figures
    axes = plot_tsdiffs(results, use_same_figure=False)
    figures = [ax.get_figure() for ax in axes]
    heading_template =  "tsdiffana plot {0}"
    headings = [heading_template.format(i) for i in range(len(figures))]

    for fig, head in zip(figures, headings):
        for_substitution['plot'] =_plot_to_svg(fig)
        for_substitution['heading'] = head
        tsdiffana_plot_text = embed_in_HTML('report_sub_template.html'
                                    ,for_substitution)
        tsdiffana_plot.append(tsdiffana_plot_text)

    return '\n'.join(tsdiffana_plot)


def generate_preproc_steps_docstring(
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