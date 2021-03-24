import string
from time import gmtime, strftime
import os
from matplotlib.pyplot import cm
import joblib
from joblib import Memory
import numpy as np
from .check_preprocessing import (plot_registration,
                                  plot_segmentation,
                                  plot_spm_motion_parameters,
                                  _plot_to_svg)
from ..time_diff import plot_tsdiffs, multi_session_time_slice_diffs
from ..io_utils import compute_mean_3D_image, sanitize_fwhm
from ..configure_spm import _configure_spm, _get_version_spm
from nilearn.plotting.html_document import HTMLDocument


PYPREPROCESS_URL = "https://github.com/neurospin/pypreprocess"
DARTEL_URL = ("https://www.fil.ion.ucl.ac.uk/spm/software/spm12/SPM12_Release_Notes.pdf")

HTML_TEMPLATE_ROOT_PATH = os.path.join(os.path.dirname(__file__),
                                           'template_reports')
SPM_DIR = _configure_spm()

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
    """ 
    Embeds components in a given HTML template.

    Parameters
    ----------
    html_template_file: .html template file 
        containing variables which would be sustituted with given components

    components_to_embed: string
        values to be substituted into the given HTML file

    """
    html_template_path = os.path.join(HTML_TEMPLATE_ROOT_PATH,
                                      html_template_file)
    with open(html_template_path) as html_file_obj:
        html_template_text = html_file_obj.read()
    string_template = string.Template(html_template_text)

    string_text = string_template.safe_substitute(**components_to_embed)

    html_file_obj.close()

    return string_text


def initialize_report(output_dir,
                    subject_name='Subject',
                    log=True,
                    filename='report',
                    prepreproc_undergone="",
                    dcm2nii=False,
                    deleteorient=False,
                    fwhm=None, anat_fwhm=None,
                    slice_timing=False,
                    realign=False,
                    coregister=False,
                    coreg_func_to_anat=False,
                    segment=False,
                    normalize=False,
                    dartel=False,
                    command_line=None,
                    has_func=True
                    ):
    """ 
    Initializes an HTML report containing the description of the 
    preprocessing steps to be implemented and the processing start time
    to be populated with visualisations for each steps.

    Parameters
    ----------
    output_dir: string
        directory to save the initialized HTML report

    log: bool, optional (default True)
        whether to initialize a log report or not

    """
    report_outfile = os.path.join(output_dir, '{}.html'.format(filename))

    report_dict = {}
    report_dict['preproc_undergone'] = generate_preproc_steps_docstring(
                                        dcm2nii=dcm2nii,
                                        deleteorient=deleteorient,
                                        slice_timing=slice_timing,
                                        realign=realign,
                                        coregister=coregister,
                                        segment=segment,
                                        normalize=normalize,
                                        fwhm=fwhm, anat_fwhm=anat_fwhm,
                                        dartel=dartel,
                                        coreg_func_to_anat=coreg_func_to_anat,
                                        prepreproc_undergone=prepreproc_undergone,
                                        has_func=has_func
                                        )
    report_dict['subject_name'] = subject_name
    report_dict['start_time'] = strftime("%d-%b-%Y %H:%M:%S", gmtime())
    report_dict['end_time'] = "STILL RUNNING..."
    report_text = embed_in_HTML('report_template.html', report_dict)
    report_HTML = HTMLDocument(report_text).save_as_html(report_outfile)

    if log:
        # create a separate HTML with all the logs
        log_outfile = os.path.join(output_dir, '{}_log.html'.format(filename))
        log_HTML = HTMLDocument("<html><body>").save_as_html(log_outfile)
        return report_outfile, log_outfile
    else:
        return report_outfile, None

def add_component(to_add_report, html_report_path, to_add_log=None,
                 html_log_path=None):
    """ 
    Appends components to the end of a given HTML report file.

    Parameters
    ----------
    to_add_report: string
        a component to be appended to a given report HTML file.
    
    html_report_path: stringlog=True
        location of the HTML report file to which the component would be
        appended.

    to_add_log: string, optional (default None)
        a log component to be added to the log report HTML file, only
        specified if a log report has been initialized.

    html_log_path: string, optional (default None)
        location of the HTML log report file to which the log component 
        would be appended, only specified if a log report has been 
        initialized.

    """
    html_file_obj = open(html_report_path, 'a')
    html_file_obj.write(to_add_report)
    html_file_obj.close()

    if html_log_path is not None:
        html_file_obj = open(html_log_path, 'a')
        html_file_obj.write('<hr/>'+to_add_log)
        html_file_obj.close()


def finalize_report(html_report_path, html_log_path=None):
    """ 
    Finalizes the report files created. Involves - adding the processing
    end time, disabling automatic page refreshing, adding closing tags at
    the bottom of the html reports and printing out report path.

    Parameters
    ----------
    html_report_path: string
        location of the HTML report file to be finalized.

    html_log_path: string, optional (default None)
        location of the HTML log report file to be finalized,
        only specified if a log report has been initialized.

    """
    html_file_obj = open(html_report_path, 'r')
    lines = html_file_obj.readlines()
    end_time = strftime("%d-%b-%Y %H:%M:%S", gmtime())
    lines[9] = "<h4 style='text-align:center'>End time: {}</h4>".format(end_time)
    del lines[4]

    with open(html_report_path, 'w') as html_file_obj:
        for line in lines:
            html_file_obj.write(line)

    html_file_obj = open(html_report_path, 'a')
    html_file_obj.write("</body>\n</html>")
    html_file_obj.close()

    if html_log_path is not None:
        html_file_obj = open(html_log_path, 'a')
        html_file_obj.write("</body>\n</html>")
        html_file_obj.close()

def generate_realignment_report(subject_data, estimated_motion, output_dir,
                                tooltip=None, log=True, report_path=None):
    """ 
    Creates visualization associated with realignment 
    and returns it as an SVG url. 

    Parameters
    ----------
    subject_data: `SubjectData` instance
       object that encapsulates the date for the subject (should have fields
       like func, anat, output_dir, etc.)

    estimated_motion: string
        location of the file containing estimated motion parameters

    output_dir: string
        directory containing all the output files
    """
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
        parameter_file=estimated_motion, lengths=lengths,
        close=True, report_path=report_path,
        title="Plot of Estimated motion for %d sessions" % len(sessions))
    for_substitution['heading'] = "Motion Correction"
    for_substitution['tooltip'] = "Motion parameters estimated during \
        motion-correction. If motion is less than half a voxel,\
        it&#x27;s generally OK. Moreover, it&#x27;s recommended\
        to include these estimated motion parameters as confounds \
        (nuissance regressors) in the the GLM."

    if log:
        log_link = "file:///"+os.path.join(output_dir, 'report_log.html#')
        for_substitution['id_link'] = for_substitution['heading'].replace(" ", "_")
        for_substitution['log'] = get_log_text(subject_data.func)
        for_substitution['log_link'] = log_link+for_substitution['id_link']
        rp_log_text = embed_in_HTML('log_sub_template.html', for_substitution)
    else:
        rp_log_text = None

    log_link_text = embed_in_HTML('log_link_template.html', for_substitution)

    for_substitution['heading'] = for_substitution['heading']+' '+log_link_text
    rp_plot_text = embed_in_HTML('report_sub_template.html', for_substitution)

    return rp_plot_text, rp_log_text


def generate_registration_report(target, source, output_dir,
                                for_substitution, report_path=None):
    """ 
    Plots target's outline on source image and returns them 
    as SVG url embedded in HTML.

    Parameters
    ----------
    target: string
        location of the .nii file for the target image

    source: string
        location of the .nii file for the target image

    output_dir: string
        directory containing all the output files

    for_substitution: dict
        a dictionary containing all the components concerning registration
        to be embedded in the report HTML files
    
    report_path: string, optional (default None)
        path to the report HTML file

    """
    reg_plot = []

    # prepare for smart caching
    qa_cache_dir = os.path.join(output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

    # plot outline (edge map) of template on the
    # normalized image
    log_link_text = embed_in_HTML('log_link_template.html', for_substitution)
    for_substitution['heading'] += ' '+log_link_text
    for_substitution['tooltip'] = "The red contours should match the background\
                                    image well. Otherwise, something might have\
                                    gone wrong. Typically things that can go\
                                    wrong include: lesions\
                                    (missing brain tissue); bad orientation\
                                    headers; non-brain tissue in anatomical\
                                    image, etc. In rare cases, it might be that\
                                    the registration algorithm simply didn&#x27;t\
                                    succeed."
    for_substitution['plot'] = qa_mem.cache(plot_registration)(
        target[0], source[0], close=True, report_path=report_path,
        title="Outline of %s on %s" % (target[1], source[1]))
    reg_plot_text = embed_in_HTML('report_sub_template.html',
                                for_substitution)
    reg_plot.append(reg_plot_text)

    # plot outline (edge map) of the normalized image
    # on the SPM MNI template
    source, target = (target, source)
    for_substitution['plot'] = qa_mem.cache(plot_registration)(
        target[0], source[0], close=True, report_path=report_path,
        title="Outline of %s on %s" % (target[1], source[1]))
    for_substitution['heading'] = ""
    for_substitution['tooltip'] = ""
    reg_plot_text = embed_in_HTML('report_sub_template.html',
                                for_substitution)
    reg_plot.append(reg_plot_text)

    return '\n'.join(reg_plot)


def generate_corregistration_report(subject_data, output_dir,
                                    coreg_func_to_anat=True, log=True,
                                    tooltip=None, report_path=None):
    """ 
    Creates plots associated with corregistration 
    and returns them as SVG url embedded in HTML.
    Calls generate_registration_plot. 
    
    Parameters
    ----------
     subject_data: `SubjectData` instance
       object that encapsulates the date for the subject (should have fields
       like func, anat, output_dir, etc.)

    output_dir: string
        directory containing all the output files
    
    report_path: string, optional (default None)
        path to the report HTML file

    """
    subject_data._set_session_ids()

    if subject_data.anat is None:
        print("Subject 'anat' field is None; nothing to do")
        return

    src, ref = subject_data.func, subject_data.anat
    src_brain, ref_brain = "mean_functional_image", "anatomical_image"
    if not coreg_func_to_anat:
        src, ref = ref, src
        src_brain, ref_brain = ref_brain, src_brain

    heading = "Corregistration %s == > %s" % (src_brain, ref_brain)

    for_substitution = {}
 


    if log:
        log_link = 'file:///'+os.path.join(output_dir, 'report_log.html#')
        for_substitution['heading'] = heading
        for_substitution['id_link'] = heading.replace(" ", "_")
        for_substitution['log'] = get_log_text(src)
        for_substitution['log_link'] = log_link+for_substitution['id_link']
        log_text = embed_in_HTML('log_sub_template.html', for_substitution)
    else:
        log_text = None

    return generate_registration_report((ref, ref_brain),
            (src, src_brain), output_dir, for_substitution,
            report_path=report_path), log_text


def generate_segmentation_report(subject_data, output_dir,
    subject_gm_file=None, subject_wm_file=None, subject_csf_file=None,
    comment="", only_native=False, tooltip=None,
    log=True, report_path=None):
    """ 
    Creates plots associated with segmentation 
    and returns them as SVG url embedded in HTML. 
    
    Parameters
    ----------
     subject_data: `SubjectData` instance
       object that encapsulates the date for the subject (should have fields
       like func, anat, output_dir, etc.)

    output_dir: string
        directory containing all the output files
    
    report_path: string, optional (default None)
        path to the report HTML file

    """
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

    log_link = 'file:///'+os.path.join(output_dir, 'report_log.html#')

    if log:
        for_substitution['heading'] = 'Segmentation'
        for_substitution['id_link'] = for_substitution['heading']
        for_substitution['log'] = get_log_text(
                                    getattr(subject_data, 'gm') or
                                    getattr(subject_data, 'wm') or
                                    getattr(subject_data, 'csf'))
        for_substitution['log_link'] = log_link+for_substitution['heading']
        log_text = embed_in_HTML('log_sub_template.html',for_substitution)
    else:
        for_substitution['log_link'] = log_link+'Segmentation'
        log_text = None

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
        for_substitution['tooltip'] = "Acronyms: TPM means Tissue Probability\
                Map; GM means Grey-Matter; WM means White-Matter; CSF means\
                Cerebro-Spinal Fuild. The TPM contours shoud match the\
                background image well. Otherwise, something might have gone\
                wrong. Typically things that can go wrong include: lesions\
                (missing brain tissue); bad orientation headers; non-brain\
                tissue in anatomical image (i.e needs brain extraction), etc.\
                In rare cases, it might be that the segmentation algorithm\
                simply didn&#x27;t succeed."
        # plot contours of template compartments on subject's brain
        if not only_native:
            for_substitution['plot']=qa_mem.cache(plot_segmentation)(
                brain, gm_filename=GM_TEMPLATE,
                wm_filename=WM_TEMPLATE, csf_filename=CSF_TEMPLATE,
                cmap=cmap, close=True, report_path=report_path,
                title=("Template GM, WM, and CSF TPM contours on "
                    "subject's %s") % _brain_name)

            log_link_text = embed_in_HTML('log_link_template.html', for_substitution)
            for_substitution['heading'] += ' '+log_link_text
            seg_plot_text = embed_in_HTML('report_sub_template.html', for_substitution)
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
                report_path=report_path, title=("%s TPM contours on "
                    "subject's %s") % (title_prefix, _brain_name))
            log_link_text = embed_in_HTML('log_link_template.html', for_substitution)
            for_substitution['heading'] += ' '+log_link_text
            if not only_native:
                for_substitution['heading'] = ""
                for_substitution['tooltip'] = ""

            seg_plot_text = embed_in_HTML('report_sub_template.html',
                                        for_substitution)
            seg_plot.append(seg_plot_text)

    if log:
        return '\n'.join(seg_plot), log_text
    else:
        return '\n'.join(seg_plot)

def generate_normalization_report(subject_data, output_dir, tooltip=None,
                                log=True, report_path=None):
    """ 
    Creates plots associated with normalization and returns them as SVG url 
    embedded in HTML. Calls generate_segmentation_report and 
    generate_registration_report. 
    
    Parameters
    ----------
     subject_data: `SubjectData` instance
       object that encapsulates the date for the subject (should have fields
       like func, anat, output_dir, etc.)

    output_dir: string
        directory containing all the output files
    
    report_path: string, optional (default None)
        path to the report HTML file

    """
    _set_templates()
    norm_plot = []
    logs = []
    subject_data._set_session_ids()

    warped_tpms = dict(
        (tpm, getattr(subject_data, tpm, None))
        for tpm in ["mwgm", "mwwm", "mwcsf"])
    segmented = warped_tpms.values().count(None) < len(warped_tpms)

    if segmented:
        plot_text = generate_segmentation_report(
                subject_data=subject_data,
                output_dir=subject_data.output_dir,
                subject_gm_file=warped_tpms["mwgm"],
                subject_wm_file=warped_tpms["mwwm"],
                subject_csf_file=warped_tpms["mwcsf"],
                comment="warped", log=False, 
                report_path=report_path)
        norm_plot.append(plot_text)
        
    for brain_name, brain, cmap in zip(
            ['anatomical_image', 'mean_functional_image'],
            [subject_data.anat, subject_data.func],
            [cm.gray, cm.nipy_spectral]):
        if not brain:
            continue

        if log:
            for_substitution = {}
            log_link = 'file:///'+os.path.join(output_dir, 'report_log.html#')
            for_substitution['heading'] = "Normalization of %s" % brain_name
            for_substitution['id_link'] = for_substitution['heading'].replace(
                                                                " ", "_")
            for_substitution['log'] = get_log_text(brain)
            for_substitution['log_link'] = log_link+for_substitution['id_link']
            log_text = embed_in_HTML('log_sub_template.html', for_substitution)
        else:
            log_text = None

        logs.append(log_text)

        if isinstance(brain, str):
            normalized = brain
        else:
            mean_normalized_img = compute_mean_3D_image(brain)
            normalized = mean_normalized_img

        norm_plot.append(generate_registration_report(
        (T1_TEMPLATE, 'template'), (normalized, brain_name),
        output_dir, for_substitution, report_path=report_path))

    return '\n'.join(norm_plot), '<hr/>'.join(logs)

def generate_tsdiffana_report(image_files, sessions, subject_id,
                            output_dir, tooltips=None):
    """ 
    Creates plots associated with tsdiffana and returns them as SVG 
    url embedded in HTML. 
    
    Parameters
    ----------
    image_files: string
        location to the functional images

    output_dir: string
        directory containing all the output files
    
    """
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
    tooltips = ["(Squared) differences across sequential volumes. A large\
        value indicates an artifact that occurred during the slice\
        acquisition, possibly related to motion.", "Average signal over each\
        volume. A large drop / peak (e.g. 1%) w.r.t the mean level indicates\
        an artefact. For example, there are  usually large values peaks in\
        the first few slices due to T2  relaxation effects, and these slices\
        are usually adviced to be discarded.", "Variance index per slice.\
        Note that acquisition artifacts can be slice-specific. Look at the\
        data if there is a peak somewhere.", "Scaled variance per slice\
        indicates slices where artifacts occur. A slice/time with large\
        variance should be eyeballed.", "Large variations should be confined\
        to vascular structures or ventricles. Large variations around the\
        brain indicate residual motion effects.", "Large variations should\
        be confined to vascular structures or ventricles. Large variations\
        around the brain indicate (uncorrected) motion effects."]

    for fig, head, tip in zip(figures, headings, tooltips):
        fig.set_rasterized(True)
        for_substitution['plot'] =_plot_to_svg(fig)
        for_substitution['heading'] = head
        for_substitution['tooltip'] = tip
        tsdiffana_plot_text = embed_in_HTML('report_sub_template.html',
                                            for_substitution)
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
    has_func=True
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

def get_log_text(nipype_output_files):
    """
    Creates properly formatted HTML log file containing the logs 
    corresponding to each of preprocessing pipeline functions calls.
    Calls get_nipype_report_filename and get_nipype_report for formatting 
    the log files.

    nipype_output_files: string
        location of the nipype log files
    """
    execution_log = get_nipype_report(get_nipype_report_filename(
        nipype_output_files))
    return execution_log


def get_nipype_report_filename(output_files_or_dir):
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


def nipype2htmlreport(nipype_report_filename):
    """
    Converts a nipype.caching report (.rst) to html.

    """
    with open(nipype_report_filename, 'r') as fd:
        return lines2breaks(fd.readlines())

def lines2breaks(lines, delimiter="\n", number_lines=False):
    """
    Converts line breaks to HTML breaks, adding `pre` tags as necessary.

    Parameters
    ----------
    lines: string delimited by delimiter, or else list of strings
        lines to format into HTML format

    delimiter: string (default '\n')
        new-line delimiter, can be (escape) characters like '\n', '\r',
        '\r\n', '\t', etc.

    number_lines: boolean (default False)
        if false, then output will be line-numbered

    Returns
    -------
    HTML-formatted string
    """
    if isinstance(lines, str):
        lines = lines.split(delimiter)
    if not number_lines:
        lines = ["%s" % line for line in lines]
        output = "<pre>%s</pre>" % "".join(lines)
    else:
        lines = ["<li>%s</li>" % line for line in lines]
        output = "<ol><pre>" + "".join(lines) + "</pre></ol>"
    return output

def pretty_time():
    """
    Returns currenct time in the format: hh:mm:ss ddd mmm yyyy.
    """
    return " ".join([time.ctime().split(" ")[i] for i in [3, 0, 2, 1, 4]])
