"""
:Module: nipype_preproc_spm_utils
:Synopsis: routine functions for SPM preprocessing business
:Author: dohmatob elvis dopgima

XXX TODO: skull-stripping and smoothing
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

# misc imports
from nisl.datasets import unzip_nii_gz

# QA imports
from check_preprocessing import *
from report_utils import *
import markup

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

    blablabla = "Generating QA reports for subject %s .." % subject_id
    dadada = "+" * len(blablabla)
    print "\r\n%s\r\n%s\r\n%s\r\n" % (dadada, blablabla, dadada)

    output = dict()
    output["plots"] = dict()
    qa_cache_dir = os.path.join(subject_output_dir, "QA")
    if not os.path.exists(qa_cache_dir):
        os.makedirs(qa_cache_dir)
    qa_mem = joblibMemory(cachedir=qa_cache_dir, verbose=5)
    report_filename = os.path.join(subject_output_dir, "_report.html")
    report = markup.page(mode='loose_html')

    report.h2(
        "CV (Coefficient of Variation) of corrected FMRI time-series")
    output["plots"] = dict()
    cv_plot_outfile1 = os.path.join(subject_output_dir, "cv_before.png")
    cv_tc_plot_outfile1 = os.path.join(subject_output_dir, "cv_tc_before.png")
    cv_tc_plot_outfile2 = os.path.join(subject_output_dir, "cv_tc_after.png")
    cv_plot_outfile2 = os.path.join(subject_output_dir, "cv_after.png")
    cv_tc_plot_outfile2 = os.path.join(subject_output_dir, "cv_tc_after.png")

    uncorrected_FMRIs = [fmri_images]
    qa_mem.cache(plot_cv_tc)(uncorrected_FMRIs, [session_id],
                             subject_id, subject_output_dir,
                             cv_plot_outfiles=[cv_plot_outfile1],
                             cv_tc_plot_outfile=cv_tc_plot_outfile1,
                             plot_diff=True,
                             title="subject %s before preproc " % subject_id)
    corrected_FMRIs = [segmented_func]
    qa_mem.cache(plot_cv_tc)(corrected_FMRIs, [session_id], subject_id,
                             subject_output_dir,
                             cv_plot_outfiles=[cv_plot_outfile2],
                             cv_tc_plot_outfile=cv_tc_plot_outfile2,
                             plot_diff=True,
                             title="subject %s after preproc " % subject_id)
    sidebyside(
        report, cv_tc_plot_outfile1,
        cv_tc_plot_outfile2)
    output["plots"]["cv_tc"] = cv_tc_plot_outfile2

    report.p("See reports for each stage below.")
    report.br()
    report.a("Motion Correction", class_='internal',
             href="realignment_report.html")
    report.a("Co-registration", class_='internal',
             href="coregistration_report.html")
    report.a("Segmentation", class_='internal',
             href="segmentation_report.html")
    report.br()

    # realignment report
    realignment_report_filename = os.path.join(subject_output_dir,
                                               "realignment_report.html")
    realignment_report = markup.page(mode="loose_html")
    realignment_report.h2(
        "Plots of estimated (rigid-body) motion in original FMRI time-series")
    motion_plot = plot_spm_motion_parameters(rp,
        subject_id=subject_id,
        title="Motion parameters of subject %s before realignment" % \
            subject_id)
    realignment_report.img(src=[motion_plot])

    nipype_report_filename = os.path.join(output_dirs['realignment'],
                                          "_report/report.rst")
    with open(nipype_report_filename, 'r') as fd:
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = markup.page(mode='loose_html')
        nipype_report.p(fd.readlines())
        open(nipype_html_report_filename, 'w').write(str(nipype_report))
        realignment_report.h2("Miscellaneous")
        realignment_report.a("nipype report", class_="internal",
                              href=nipype_html_report_filename)

    with open(realignment_report_filename, 'w') as fd:
        fd.write(str(realignment_report))
        fd.close()

    output["plots"]["realignment_parameters"] = motion_plot

    # coreg report
    coregistration_report_filename = os.path.join(
        subject_output_dir,
        "coregistration_report.html")
    coregistration_report = markup.page(mode='loose_html')
    coregistration_report.h1(
        "Coregistration for subject %s (mean functional -> anat)" % subject_id)
    overlaps_before = []
    overlaps_after = []
    overlap_plot = os.path.join(output_dirs["coregistration"],
                                "overlap_func_on_anat_before.png")
    qa_mem.cache(plot_registration)(realign_result.outputs.mean_image,
                                    anat_image,
                                    output_filename=overlap_plot,
                                    title="Before coregistration: overlap of \
(mean) functional on anatomical for subject %s" % subject_id)

    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(output_dirs["coregistration"],
                                "overlap_anat_on_func_before.png")
    qa_mem.cache(plot_registration)(anat_image,
                                    realign_result.outputs.mean_image,
                                    output_filename=overlap_plot,
                                    title="After coregistration: overlap of \
(mean) functional on anatomical for subject %s" % subject_id)

    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(output_dirs["coregistration"],
                                "overlap_func_on_anat_after.png")
    qa_mem.cache(plot_registration)(
        coreg_result.outputs.coregistered_source,
        anat_image,
        output_filename=overlap_plot,
        )
    overlaps_after.append(overlap_plot)
    overlap_plot = os.path.join(output_dirs["coregistration"],
                                "overlap_anat_on_func_after.png")
    qa_mem.cache(plot_registration)(anat_image,
                                 coreg_result.outputs.coregistered_source,
                                 output_filename=overlap_plot,
                                 )
    overlaps_after.append(overlap_plot)
    sidebyside(coregistration_report, overlaps_before, overlaps_after)

    nipype_report_filename = os.path.join(output_dirs['coregistration'],
                                          "_report/report.rst")
    with open(nipype_report_filename, 'r') as fd:
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = markup.page(mode='loose_html')
        nipype_report.p(fd.readlines())
        open(nipype_html_report_filename, 'w').write(str(nipype_report))
        coregistration_report.h2("Miscellaneous")
        coregistration_report.a("nipype report", class_="internal",
                              href=nipype_html_report_filename)
    with open(coregistration_report_filename, 'w') as fd:
        fd.write(str(coregistration_report))
        fd.close()

    output["plots"]["coregistration"] = overlap_plot

    # segment report
    tmp = os.path.dirname(segmented_mean_func)
    segmentation_report_filename = os.path.join(
        subject_output_dir,
        "segmentation_report.html")
    segmentation_report = markup.page(mode='loose_html')
    segmentation_report.h1("Segmentation report for subject: %s" % subject_id)

    before_segmentation = os.path.join(tmp, "before_segmentation.png")
    qa_mem.cache(plot_segmentation)(realign_result.outputs.mean_image,
                                    GM_TEMPLATE,
                                    WM_TEMPLATE,
                                    CSF_TEMPLATE,
                                    output_filename=before_segmentation,
                                    title="Before segmentation: GM, WM, and\
 CSF contour maps of subject %s's mean functional" % subject_id)

    after_segmentation = os.path.join(tmp, "after_segmentation.png")
    after_segmentation_summary = os.path.join(tmp,
                                              "after_segmentation_summary.png")
    qa_mem.cache(plot_segmentation)(segmented_mean_func,
                                    segment_result.outputs.modulated_gm_image,
                                    segment_result.outputs.modulated_wm_image,
                                    segment_result.outputs.modulated_csf_image,
                                    output_filename=after_segmentation_summary,
                                    slicer='z',
                                    title="subject: %s" % subject_id)
    qa_mem.cache(plot_segmentation)(segmented_mean_func,
                                    segment_result.outputs.modulated_gm_image,
                                    segment_result.outputs.modulated_wm_image,
                                    segment_result.outputs.modulated_csf_image,
                                    output_filename=after_segmentation,
                                    title="After segmentation: GM, WM, and CSF \
contour maps of subject %s's mean functional" % subject_id)

    sidebyside(segmentation_report, before_segmentation, after_segmentation)

    nipype_report_filename = os.path.join(tmp,
                                          "_report/report.rst")
    with open(nipype_report_filename, 'r') as fd:
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = markup.page(mode='loose_html')
        nipype_report.p(fd.readlines())
        open(nipype_html_report_filename, 'w').write(str(nipype_report))
        segmentation_report.h2("Miscellaneous")
        segmentation_report.a("nipype report", class_="internal",
                              href=nipype_html_report_filename)

    with open(segmentation_report_filename, 'w') as fd:
        fd.write(str(segmentation_report))
        fd.close()

    with open(report_filename, 'w') as fd:
        fd.write(str(report))
        fd.close()

    output["plots"]["segmentation"] = after_segmentation
    output["plots"]["segmentation_summary"] = after_segmentation_summary

    output["report"] = report_filename

    return subject_id, session_id, output
