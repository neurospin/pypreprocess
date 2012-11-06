"""
:Module: nipype_preproc_spm_utils
:Synopsis: routine functions for SPM preprocessing business
:Author: dohmatob elvis dopgima

XXX: Add skull-stripping and smoothing
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

# MISC
COREG_TOLERANCE = [0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01,
                   0.01, 0.01, 0.001, 0.001, 0.001]

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
    coreg_result = coreg(target=realign_result.outputs.mean_image,
                         source=anat_image,
                         # cost_function="nmi",
                         # tolerance=COREG_TOLERANCE,
                         # fwhm=7,
                         # separation=2,
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
    segment_result = segment(data=coreg_result.outputs.coregistered_source,
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
        apply_to_files=coreg_result.outputs.coregistered_source,
        jobtype='write',
        # write_voxel_sizes=[1, 1, 1]
        )

    # generate html report (for QA)
    report_filename = os.path.join(subject_output_dir, "_report.html")
    report = markup.page(mode='strict_html')

    report.h1(
        "Plots of estimated (rigid-body) motion in original FMRI time-series")
    motion_plot = plot_spm_motion_parameters(
        os.path.join(output_dirs["realignment"], "rp_bet_lfo.txt"),
        subject_id=subject_id,
        title="before realignment")
    report.img(src=[motion_plot])

    report.h1(
        "Coregistration (anat -> mean functional)")
    report.h2("Before coregistration")
    overlap_plot1 = os.path.join(output_dirs["coregistration"],
                                 "overlap_anat_on_func_before.png")
    plot_coregistration(realign_result.outputs.mean_image,
                        anat_image,
                        plot_outfile=overlap_plot1)
    overlap_plot2 = os.path.join(output_dirs["coregistration"],
                                 "overlap_func_on_anat_before.png")
    plot_coregistration(
                        anat_image,
                        realign_result.outputs.mean_image,
                        plot_outfile=overlap_plot2)
    report.img(src=[overlap_plot1, overlap_plot2])
    report.h2("After coregistration")
    overlap_plot1 = os.path.join(output_dirs["coregistration"],
                                 "overlap_anat_on_func_after.png")
    plot_coregistration(realign_result.outputs.mean_image,
                        coreg_result.outputs.coregistered_source,
                        plot_outfile=overlap_plot1)
    overlap_plot2 = os.path.join(output_dirs["coregistration"],
                                 "overlap_func_on_anat_after.png")
    plot_coregistration(
                        coreg_result.outputs.coregistered_source,
                        realign_result.outputs.mean_image,
                        plot_outfile=overlap_plot2)
    report.img(src=[overlap_plot1, overlap_plot2])
    report.br()
    report.br()

    report.h1(
        "CV (Coefficient of Variation) of corrected FMRI time-series")
    cv_tc_plot_outfile1 = os.path.join(subject_output_dir, "cv_tc_before.png")
    uncorrected_FMRIs = glob.glob(
        os.path.join(subject_output_dir,
                     "func/lfo.nii"))
    plot_cv_tc(uncorrected_FMRIs, [session_id], subject_id,
               plot_outfile=cv_tc_plot_outfile1,
               title="before preproc")
    cv_tc_plot_outfile2 = os.path.join(subject_output_dir, "cv_tc_after.png")
    corrected_FMRIs = glob.glob(
        os.path.join(subject_output_dir,
                     "wrbet_lfo.nii"))
    plot_cv_tc(corrected_FMRIs, [session_id], subject_id,
               plot_outfile=cv_tc_plot_outfile2,
               title="after preproc")
    report.img(src=[cv_tc_plot_outfile1, cv_tc_plot_outfile2])
    report.br()
    report.br()


    with open(report_filename, 'w') as fd:
        fd.write(str(report))
        fd.close()

    return subject_id, session_id, output_dirs
