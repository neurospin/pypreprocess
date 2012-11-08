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
import re

# import useful interfaces from nipype
from nipype.caching import Memory
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab
import nipype.interfaces.fsl as fsl

# misc imports
from nisl.datasets import unzip_nii_gz

# QA imports
from check_preprocessing import *
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

# # fsl output_type
# fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# fsl BET cmd prefix
bet_cmd_prefix = "fsl4.1-"

# set templates
T1_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'templates/T1.nii')
GM_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/grey.nii')
WM_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/white.nii')
CSF_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/csf.nii')


def sidebyside(report, img1, img2):
    if not type(img1) is list:
        img1 = [img1]
    if not type(img2) is list:
        img2 = [img2]

    report.center()
    report.table(border="0", width="100%", cellspacing="1")
    report.tr()
    report.td()
    report.img(src=img1, width="100%")
    report.td.close()
    report.td()
    report.img(src=img2, width="100%")
    report.td.close()
    report.tr.close()
    report.table.close()
    report.center.close()


def do_subject_preproc(subject_id,
                       subject_output_dir,
                       anat_image,
                       fmri_images,
                       session_id=None,
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
    mem = Memory(base_dir=cache_dir)

    output_dirs = dict()

    # Brain Extraction
    fmri_dir = os.path.dirname(fmri_images)
    bet_out_file = os.path.join(fmri_dir,
                                "bet_" + os.path.basename(fmri_images))
    if not os.path.exists(bet_out_file):
        bet = fsl.BET(
            in_file=fmri_images,
            out_file=bet_out_file)
        bet._cmd = bet_cmd_prefix + bet._cmd
        bet.inputs.functional = True
        bet.run()
        unzip_nii_gz(fmri_dir)
    #  motion correction
    realign = mem.cache(spm.Realign)
    realign_result = realign(in_files=bet_out_file,
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
                         jobtype='estimate')
    output_dirs["coregistration"] = os.path.dirname(
        coreg_result.outputs.coregistered_source)

    # # learn the deformation on T1 MRI without segmentation
    # normalize = mem.cache(spm.Normalize)
    # norm_result = normalize(source=coreg_result.outputs.coregistered_source,
    #                         template=T1_TEMPLATE)

    # # deform FRMI images unto T1 template
    # norm_apply = normalize(
    #     parameter_file=norm_result.outputs.normalization_parameters,
    #     apply_to_files=realign_result.outputs.realigned_files,
    #     jobtype='write',
    #     write_voxel_sizes=[3, 3, 3])

    # wfmri = norm_apply.outputs.normalized_files
    # # if type(wfmri) is str:
    # #     wfmri = [wfmri]
    # # for wf in wfmri:
    # #     shutil.copyfile(wf, os.path.join(subject_output_dir,
    # #                                      os.path.basename(wf)))

    # # deform anat image unto T1 template
    # norm_apply = normalize(
    #     parameter_file=norm_result.outputs.normalization_parameters,
    #     apply_to_files=coreg_result.outputs.coregistered_source,
    #     jobtype='write',
    #     write_voxel_sizes=[1, 1, 1])

    # wanat = norm_apply.outputs.normalized_files
    # # shutil.copyfile(wanat, os.path.join(t1_dir, os.path.basename(wanat)))

    #  alternative: Segmentation & normalization
    normalize = mem.cache(spm.Normalize)
    segment = mem.cache(spm.Segment)
    segment_result = segment(data=anat_image,
                             gm_output_type=[True, True, True],
                             wm_output_type=[True, True, True],
                             csf_output_type=[True, True, True],
                             tissue_prob_maps=[GM_TEMPLATE,
                                               WM_TEMPLATE, CSF_TEMPLATE])

    # segment the realigned FMRI
    norm_apply = normalize(
        parameter_file=segment_result.outputs.transformation_mat,
        apply_to_files=realign_result.outputs.realigned_files,
        jobtype='write',
        # write_voxel_sizes=[3, 3, 3]
        )

    wfmri = norm_apply.outputs.normalized_files
    shutil.copyfile(wfmri, os.path.join(subject_output_dir,
                                        os.path.basename(wfmri)))

    # segment the coregistered anatomical
    norm_apply = normalize(
        parameter_file=segment_result.outputs.transformation_mat,
        apply_to_files=anat_image,
        jobtype='write',
        # write_voxel_sizes=[1, 1, 1]
        )

    segmented_anat = norm_apply.outputs.normalized_files

    # generate html report (for QA)
    report_filename = os.path.join(subject_output_dir, "_report.html")
    report = markup.page(mode='loose_html')

    report.h2(
        "CV (Coefficient of Variation) of corrected FMRI time-series")
    cv_tc_plot_outfile1 = os.path.join(subject_output_dir, "cv_tc_before.png")
    cv_tc_plot_outfile2 = os.path.join(subject_output_dir, "cv_tc_after.png")
    cv_tc_plot_outfile3 = os.path.join(subject_output_dir,
                                       "cv_tc_diff_before.png")
    cv_tc_plot_outfile4 = os.path.join(subject_output_dir,
                                       "cv_tc_diff_after.png")
    uncorrected_FMRIs = glob.glob(
        os.path.join(subject_output_dir,
                     "func/lfo.nii"))
    plot_cv_tc(uncorrected_FMRIs, [session_id], subject_id,
               plot_outfile=cv_tc_plot_outfile1,
               title="before preproc",
               plot_cv_tc_diff=cv_tc_plot_outfile3)
    corrected_FMRIs = glob.glob(
        os.path.join(subject_output_dir,
                     "wrbet_lfo.nii"))
    plot_cv_tc(corrected_FMRIs, [session_id], subject_id,
               plot_outfile=cv_tc_plot_outfile2,
               title="after preproc",
               plot_cv_tc_diff=cv_tc_plot_outfile4)

    uncorrected_FMRIs = glob.glob(
        os.path.join(subject_output_dir,
                     "func/lfo.nii"))
    corrected_FMRIs = glob.glob(
        os.path.join(subject_output_dir,
                     "wrbet_lfo.nii"))

    sidebyside(report, [cv_tc_plot_outfile1, cv_tc_plot_outfile3],
               [cv_tc_plot_outfile2, cv_tc_plot_outfile4])
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
    motion_plot = plot_spm_motion_parameters(
        os.path.join(output_dirs["realignment"], "rp_bet_lfo.txt"),
        subject_id=subject_id,
        title="before realignment")
    realignment_report.img(src=[motion_plot])

    nipype_report_filename = os.path.join(output_dirs['realignment'],
                                          "_report/report.rst")
    with open(nipype_report_filename, 'r') as fd:
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = markup.page(mode='loose_html')
        nipype_report.p(fd.readlines())
        open(nipype_html_report_filename, 'w').write(str(nipype_report))
        realignment_report.a("nipype report", class_="internal",
                              href=nipype_html_report_filename)

    with open(realignment_report_filename, 'w') as fd:
        fd.write(str(realignment_report))
        fd.close()

    # coreg report
    coregistration_report_filename = os.path.join(
        subject_output_dir,
        "coregistration_report.html")
    coregistration_report = markup.page(mode='loose_html')
    coregistration_report.h2(
        "Coregistration (mean functional -> anat)")
    overlaps_before = []
    overlaps_after = []
    overlap_plot = os.path.join(output_dirs["coregistration"],
                                "overlap_func_on_anat_before.png")
    plot_coregistration(realign_result.outputs.mean_image,
                        anat_image,
                        plot_outfile=overlap_plot,
                        )
    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(output_dirs["coregistration"],
                                "overlap_anat_on_func_before.png")

    plot_coregistration(anat_image,
                        realign_result.outputs.mean_image,
                        plot_outfile=overlap_plot,
                        )
    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(output_dirs["coregistration"],
                                "overlap_func_on_anat_after.png")
    plot_coregistration(coreg_result.outputs.coregistered_source,
                        anat_image,
                        plot_outfile=overlap_plot,
                        )
    overlaps_after.append(overlap_plot)
    overlap_plot = os.path.join(output_dirs["coregistration"],
                                "overlap_anat_on_func_after.png")
    plot_coregistration(anat_image,
                        coreg_result.outputs.coregistered_source,
                        plot_outfile=overlap_plot,
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
        coregistration_report.a("nipype report", class_="internal",
                              href=nipype_html_report_filename)
    with open(coregistration_report_filename, 'w') as fd:
        fd.write(str(coregistration_report))
        fd.close()

    # segment report
    tmp = os.path.dirname(segmented_anat)
    segmentation_report_filename = os.path.join(
        subject_output_dir,
        "segmentation_report.html")
    segmentation_report = markup.page(mode='loose_html')

    segmentation_report.p("Left column are plots before segmentaion \
and right column are plots after; top plots are plots of template\
 over segmented anat, whiltst bottom plots are the reverse.")

    segmentation_report.h2(
        "anat -> grey matter")
    overlaps_before = []
    overlaps_after = []
    overlap_plot = os.path.join(tmp,
                                "overlap_anat_on_gm_before.png")
    plot_coregistration(segment_result.outputs.modulated_gm_image,
                        anat_image,
                        plot_outfile=overlap_plot,
                        )
    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_gm_on_anat_before.png")

    plot_coregistration(
                        anat_image,
                        segment_result.outputs.modulated_gm_image,
                        plot_outfile=overlap_plot,
                        )
    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_anat_on_gm_after.png")
    plot_coregistration(
        segment_result.outputs.modulated_gm_image,
        segmented_anat,
        plot_outfile=overlap_plot,
        )
    overlaps_after.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_gm_on_anat_after.png")
    plot_coregistration(
        segmented_anat,
        segment_result.outputs.modulated_gm_image,
        plot_outfile=overlap_plot,
        )
    overlaps_after.append(overlap_plot)
    sidebyside(segmentation_report, overlaps_before, overlaps_after)

    segmentation_report.h2(
        "anat -> white matter")
    overlaps_before = []
    overlaps_after = []
    overlap_plot = os.path.join(tmp,
                                "overlap_anat_on_wm_before.png")
    plot_coregistration(WM_TEMPLATE,
                        anat_image,
                        plot_outfile=overlap_plot,
                        )
    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_wm_on_anat_before.png")

    plot_coregistration(
                        anat_image,
                        WM_TEMPLATE,
                        plot_outfile=overlap_plot,
                        )
    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_anat_on_wm_after.png")
    plot_coregistration(
        WM_TEMPLATE,
        segmented_anat,
        plot_outfile=overlap_plot,
        )
    overlaps_after.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_wm_on_anat_after.png")
    plot_coregistration(
        segmented_anat,
        WM_TEMPLATE,
        plot_outfile=overlap_plot,
        )
    overlaps_after.append(overlap_plot)
    sidebyside(segmentation_report, overlaps_before, overlaps_after)

    segmentation_report.h2(
        "anat -> csf")
    overlaps_before = []
    overlaps_after = []
    overlap_plot = os.path.join(tmp,
                                "overlap_anat_on_csf_before.png")
    plot_coregistration(CSF_TEMPLATE,
                        anat_image,
                        plot_outfile=overlap_plot,
                        )
    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_csf_on_anat_before.png")

    plot_coregistration(
                        anat_image,
                        CSF_TEMPLATE,
                        plot_outfile=overlap_plot,
                        )
    overlaps_before.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_anat_on_csf_after.png")
    plot_coregistration(
        CSF_TEMPLATE,
        segmented_anat,
        plot_outfile=overlap_plot,
        )
    overlaps_after.append(overlap_plot)
    overlap_plot = os.path.join(tmp,
                                "overlap_csf_on_anat_after.png")
    plot_coregistration(
        segmented_anat,
        CSF_TEMPLATE,
        plot_outfile=overlap_plot,
        )
    overlaps_after.append(overlap_plot)
    sidebyside(segmentation_report, overlaps_before, overlaps_after)

    nipype_report_filename = os.path.join(tmp,
                                          "_report/report.rst")
    with open(nipype_report_filename, 'r') as fd:
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = markup.page(mode='loose_html')
        nipype_report.p(fd.readlines())
        open(nipype_html_report_filename, 'w').write(str(nipype_report))
        segmentation_report.a("nipype report", class_="internal",
                              href=nipype_html_report_filename)

    with open(segmentation_report_filename, 'w') as fd:
        fd.write(str(segmentation_report))
        fd.close()

    with open(report_filename, 'w') as fd:
        fd.write(str(report))
        fd.close()

    return subject_id, session_id, output_dirs
