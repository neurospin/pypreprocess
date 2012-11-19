"""
:Module: nipype_preproc_spm_utils
:Synopsis: routine functions for SPM preprocessing business
:Author: dohmatob elvis dopgima

XXX TODO: document the code!
XXX TODO: re-factor the code!
"""

# standard imports
import os
import shutil

# imports for caching (yeah, we aint got time to loose!)
from nipype.caching import Memory as nipypeMemory
from joblib import Memory as joblibMemory

# spm and matlab imports
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab

# parallelism imports
from joblib import Parallel, delayed
from multiprocessing import cpu_count

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
MATLAB_SPM_DIR = '/i2bm/local/spm8'
if 'MATLAB_SPM_DIR' in os.environ:
    MATLAB_SPM_DIR = os.environ['MATLAB_SPM_DIR']
assert os.path.exists(MATLAB_SPM_DIR), \
    "nipype_preproc_smp_utils: MATLAB_SPM_DIR: %s,\
 doesn't exist; you need to export MATLAB_SPM_DIR" % MATLAB_SPM_DIR
matlab.MatlabCommand.set_default_paths(MATLAB_SPM_DIR)

# set job count
N_JOBS = -1
if 'N_JOBS' in os.environ:
    N_JOBS = int(os.environ['N_JOBS'])

# set templates
T1_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'templates/T1.nii')
GM_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/grey.nii')
WM_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/white.nii')
CSF_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/csf.nii')


def do_subject_preproc(subject_id,
                       subject_output_dir,
                       anat_image,
                       fmri_images,
                       session_id="UNKNOWN",
                       do_report=True,
                       **kwargs):
    """
    Function preprocessing data for a single subject.

    """
    # prepare for smart-caching
    t1_dir = os.path.join(subject_output_dir, 't1')
    realign_ouput_dir = os.path.join(subject_output_dir, "realign")
    if not os.path.exists(realign_ouput_dir):
        os.makedirs(realign_ouput_dir)
    if not os.path.exists(t1_dir):
        os.makedirs(t1_dir)
    cache_dir = os.path.join(subject_output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = nipypeMemory(base_dir=cache_dir)

    output_dirs = dict()

    #  motion correction
    realign = mem.cache(spm.Realign)
    realign_result = realign(in_files=fmri_images,
                             register_to_mean=True,
                             mfile=True,
                             jobtype='estwrite')
    rp = realign_result.outputs.realignment_parameters

    shutil.copyfile(rp,
                    os.path.join(realign_ouput_dir, os.path.basename(rp)))
    output_dirs["realignment"] = os.path.dirname(rp)

    # co-registration of functional against structural (anatomical)
    coreg = mem.cache(spm.Coregister)
    coreg_result = coreg(target=anat_image,
                         source=realign_result.outputs.mean_image,
                         apply_to_files=realign_result.outputs.realigned_files,
                         jobtype='estimate')
    output_dirs["coregistration"] = os.path.dirname(
        coreg_result.outputs.coregistered_source)

    # Segmentation & normalization
    normalize = mem.cache(spm.Normalize)
    segment = mem.cache(spm.Segment)
    segment_result = segment(data=anat_image,
                             gm_output_type=[True, True, True],
                             wm_output_type=[True, True, True],
                             csf_output_type=[True, True, True],
                             tissue_prob_maps=[GM_TEMPLATE,
                                               WM_TEMPLATE, CSF_TEMPLATE])

    # segment the coregistered anatomical
    norm_apply = normalize(
        parameter_file=segment_result.outputs.transformation_mat,
        apply_to_files=anat_image,
        jobtype='write',
        # write_voxel_sizes=[1, 1, 1]
        )
    segmented_anat = norm_apply.outputs.normalized_files
    shutil.copyfile(segmented_anat, os.path.join(
            subject_output_dir,
            os.path.basename(segmented_anat)))

    # segment the mean FMRI
    norm_apply = normalize(
        parameter_file=segment_result.outputs.transformation_mat,
        apply_to_files=coreg_result.outputs.coregistered_source,
        jobtype='write',
        )
    segmented_mean_func = norm_apply.outputs.normalized_files
    shutil.copyfile(segmented_mean_func, os.path.join(
            subject_output_dir,
            os.path.basename(segmented_mean_func)))

    # segment the FMRI
    norm_apply = normalize(
        parameter_file=segment_result.outputs.transformation_mat,
        apply_to_files=coreg_result.outputs.coregistered_files,
        jobtype='write',
        )
    segmented_func = norm_apply.outputs.normalized_files
    shutil.copyfile(segmented_func, os.path.join(
            subject_output_dir,
            os.path.basename(segmented_func)))

    # generate html report (for QA)
    if do_report:
        import check_preprocessing
        import tempita
        import reporter
        import time

        report_filename = os.path.join(subject_output_dir, "_report.html")
        plots_gallery = []

        blablabla = "Generating QA reports for subject %s .." % subject_id
        dadada = "+" * len(blablabla)
        print "\r\n%s\r\n%s\r\n%s\r\n" % (dadada, blablabla, dadada)

        output = dict()
        output["plots"] = dict()
        qa_cache_dir = os.path.join(subject_output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblibMemory(cachedir=qa_cache_dir, verbose=5)

        cv_tc_plot_before = os.path.join(subject_output_dir,
                                         "cv_tc_before.png")
        cv_tc_plot_after = os.path.join(subject_output_dir, "cv_tc_after.png")

        uncorrected_FMRIs = [fmri_images]
        qa_mem.cache(check_preprocessing.plot_cv_tc)(
            uncorrected_FMRIs, [session_id],
            subject_id, subject_output_dir,
            cv_tc_plot_outfile=cv_tc_plot_before,
            plot_diff=True,
            title="subject %s before preproc " % subject_id)

        corrected_FMRIs = [segmented_func]
        qa_mem.cache(check_preprocessing.plot_cv_tc)(
            corrected_FMRIs, [session_id], subject_id,
            subject_output_dir,
            cv_tc_plot_outfile=cv_tc_plot_after,
            plot_diff=True,
            title="subject %s after preproc " % subject_id)

        # plots_gallery.append((cv_tc_plot_before, 'subject: %s' % subject_id,
        #                       cv_tc_plot_after, ""))

        # realignment report
        motion_plot = check_preprocessing.plot_spm_motion_parameters(
        rp,
        subject_id=subject_id,
        title="Plot of motion parameters before realignment")

        nipype_report_filename = os.path.join(output_dirs['realignment'],
                                              "_report/report.rst")
        with open(nipype_report_filename, 'r') as fd:
            nipype_html_report_filename = nipype_report_filename + '.html'
            nipype_report = reporter.nipype2htmlreport(nipype_report_filename)
            open(nipype_html_report_filename, 'w').write(str(nipype_report))

        output["plots"]["motion"] = motion_plot
        plots_gallery.append((motion_plot, 'subject: %s' % subject_id,
                              motion_plot, nipype_html_report_filename))

        # coreg report
        anat_on_func_overlap_summary = os.path.join(
            output_dirs["coregistration"],
            "anat_on_func_overlap_summary.png")
        qa_mem.cache(check_preprocessing.plot_registration)(
            anat_image,
            coreg_result.outputs.coregistered_source,
            output_filename=anat_on_func_overlap_summary,
            title="%s: Anat on functional" % subject_id,
            slicer='z')

        anat_on_func_overlap = os.path.join(
            output_dirs["coregistration"],
            "anat_on_func_overlap.png")
        qa_mem.cache(check_preprocessing.plot_registration)(
            anat_image,
            coreg_result.outputs.coregistered_source,
            output_filename=anat_on_func_overlap,
            title="Overlap of anatomical on \
(mean) function for subject %s" % subject_id)

        nipype_report_filename = os.path.join(output_dirs['coregistration'],
                                              "_report/report.rst")
        with open(nipype_report_filename, 'r') as fd:
            nipype_html_report_filename = nipype_report_filename + '.html'
            nipype_report = reporter.nipype2htmlreport(
                nipype_report_filename)
            open(nipype_html_report_filename, 'w').write(str(nipype_report))

        plots_gallery.append(
            (anat_on_func_overlap,
             'subject: %s' % subject_id,
             anat_on_func_overlap_summary, nipype_html_report_filename))

        # segment report
        output["plots"]["segmentation"] = dict()
        tmp = os.path.dirname(segmented_mean_func)

        # plot contours of template TPMs on subjects (mean) functional
        output["plots"]["segmentation"]["template_tpms"] = dict()
        template_tpms_contours = os.path.join(
            tmp,
            "template_tmps_contours.png")
        template_tpms_contours_summary = os.path.join(
            tmp,
            "template_tpms_contours_summary.png")

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            segmented_mean_func,
            GM_TEMPLATE,
            WM_TEMPLATE,
            CSF_TEMPLATE,
            output_filename=template_tpms_contours_summary,
            slicer='z',
            title="%s: template TPMs" % subject_id)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            segmented_mean_func,
            GM_TEMPLATE,
            WM_TEMPLATE,
            CSF_TEMPLATE,
            output_filename=template_tpms_contours,
            title="Template GM, WM, and CSF contours of subject %s's \
(mean) functional" % subject_id)

        plots_gallery.append((template_tpms_contours,
                              "subject: %s" % subject_id,
                              template_tpms_contours_summary,
                              nipype_html_report_filename))

        output["plots"]["segmentation"]["template_tpms"]["full"] = \
            template_tpms_contours
        output["plots"]["segmentation"]["template_tpms"]["summary"] = \
template_tpms_contours_summary

        # plot contours of subject's TPMs on subjects (mean) function
        output["plots"]["segmentation"]["subject_tpms"] = dict()
        subject_tpms_contours = os.path.join(
            tmp,
            "subject_tmps_contours.png")
        subject_tpms_contours_summary = os.path.join(
            tmp,
            "subject_tpms_contours_summary.png")

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            segmented_mean_func,
            segment_result.outputs.modulated_gm_image,
            segment_result.outputs.modulated_wm_image,
            segment_result.outputs.modulated_csf_image,
            output_filename=subject_tpms_contours_summary,
            slicer='z',
            title="%s: subject TPMs" % subject_id)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            segmented_mean_func,
            segment_result.outputs.modulated_gm_image,
            segment_result.outputs.modulated_wm_image,
            segment_result.outputs.modulated_csf_image,
            output_filename=subject_tpms_contours,
            title="Subject %s's GM, WM, and CSF contours on their (mean)\
 functional" % subject_id)

        plots_gallery.append((subject_tpms_contours,
                              "subject: %s" % subject_id,
                              subject_tpms_contours_summary,
                              nipype_html_report_filename))

        output["plots"]["segmentation"]["subject_tpms"]["full"] = \
            subject_tpms_contours
        output["plots"]["segmentation"]["subject_tpms"]["summary"] = \
subject_tpms_contours_summary

        nipype_report_filename = os.path.join(tmp,
                                              "_report/report.rst")
        with open(nipype_report_filename, 'r') as fd:
            nipype_html_report_filename = nipype_report_filename + '.html'
            nipype_report = reporter.nipype2htmlreport(
                nipype_report_filename)
            open(nipype_html_report_filename, 'w').write(str(nipype_report))

        report = reporter.BASE_PREPROC_REPORT_HTML_TEMPLATE.substitute(
            now=time.ctime(), plots_gallery=plots_gallery,
            height=400, report_name="Report for subject %s" % subject_id)

        print ">" * 80 + "BEGIN HTML"
        print report
        print "<" * 80 + "END HTML"

        with open(report_filename, 'w') as fd:
            fd.write(str(report))
            fd.close()

        output["report"] = report_filename

        return subject_id, session_id, output


def subject_callback(args):
    subject_id, subject_dir, anat_image, fmri_images, session_id = args
    return do_subject_preproc(subject_id, subject_dir, anat_image,
                              fmri_images, session_id=session_id)


def do_group_preproc(subjects,
                     do_report=True,
                     dataset_description=None,
                     report_filename=None,
                     **kwargs):

    results = Parallel(n_jobs=N_JOBS)(delayed(subject_callback)(args) \
                                      for args in subjects)

    # generate html report (for QA)
    if do_report:
        import tempita
        import reporter
        import time
        blablabla = "Generating QA report for %d subjects .." % len(results)
        dadada = "+" * len(blablabla)
        print "\r\n%s\r\n%s\r\n%s\r\n" % (dadada, blablabla, dadada)

        tmpl = reporter.BASE_PREPROC_REPORT_HTML_TEMPLATE
        plots_gallery = list()

        for subject_id, session_id, output in results:
            full_plot = \
                output['plots']['segmentation']["template_tpms"]["full"]
            title = 'subject: %s' % subject_id
            summary_plot = \
                output['plots']["segmentation"]["template_tpms"]["summary"]
            redirect_url = output["report"]
            plots_gallery.append((full_plot, title, summary_plot,
                                  redirect_url))

        report  = tmpl.substitute(
            now=time.ctime(),
            plots_gallery=plots_gallery,
            dataset_description=tempita.html(
                reporter.lines2breaks(dataset_description)))

        print ">" * 80 + "BEGIN HTML"
        print report
        print "<" * 80 + "END HTML\r\n"

        if not report_filename is None:
            with open(report_filename, 'w') as fd:
                fd.write(str(report))
                fd.close()
                print "HTML report written to %s" % report_filename

        print "\r\nDone."
