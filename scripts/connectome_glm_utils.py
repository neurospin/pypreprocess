"""
:Author: DOHMATOB Elvis Dopgima

"""

import os
import numpy as np
import nibabel
from nipy.modalities.fmri.glm import FMRILinearModel
from pypreprocess.nipype_preproc_spm_utils_bis import (SubjectData,
                                                       do_subject_preproc)
from pypreprocess.fsl_to_nipy import (read_design_fsl_design_file,
                                      make_dmtx_from_timing_files,
                                      _insert_directory_in_file_name)
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.reporting.base_reporter import (ProgressReport,
                                                  pretty_time)


def run_suject_level1_glm(subject_data_dir, subject_output_dir,
                          tr=.72,
                          task="MOTOR",
                          fwhm=0.,
                          hrf_model="Canonical with Derivative",
                          drift_model="Cosine",
                          hfcut=100,
                          regress_motion=True,
                          motion_params_file_pattern="Movement_Regressors.txt"
                          ):
    # sanitize subject_output_dir
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)

    # chronometry
    stats_start_time = pretty_time()

    # merged lists
    paradigms = []
    design_matrices = []
    fmri_files = []
    n_scans = []
    for direction in ['LR', 'RL']:
        # importat maps
        z_maps = {}
        effects_maps = {}

        # subject_id
        subject_id = os.path.basename(subject_data_dir)

        # glob the design file
        design_file = os.path.join(subject_data_dir, "tfMRI_%s_%s" % (
                task, direction),
                                   "tfMRI_%s_%s_hp200_s4_level1.fsf" % (
                task, direction))

        # glog motion parameter files
        add_regs_file = None
        if regress_motion:
            add_regs_file = os.path.join(subject_data_dir,
                                          "tfMRI_%s_%s" % (task, direction),
                                          motion_params_file_pattern)

        # glob the preprocessed 4D fmri file
        fmri_file = os.path.join(subject_data_dir, "tfMRI_%s_%s" % (
                task, direction), "tfMRI_MOTOR_%s.nii.gz" % direction)
        fmri_files.append(fmri_file)

        # read the experimental setup
        print "Reading experimental setup from %s ..." % design_file
        fsl_condition_ids, timing_files, contrast_ids, contrast_values = \
            read_design_fsl_design_file(design_file)
        print "... done.\r\n"

        # fix timing filenames
        timing_files = _insert_directory_in_file_name(
            timing_files, "tfMRI_%s_%s" % (task, direction), 1)

        # make design matrix
        _n_scans = nibabel.load(fmri_file).shape[-1]
        n_scans.append(_n_scans)
        design_matrix, paradigm, frametimes = make_dmtx_from_timing_files(
            timing_files, fsl_condition_ids, n_scans=_n_scans, tr=tr,
            hrf_model=hrf_model, drift_model=drift_model, hfcut=hfcut,
            add_regs_file=add_regs_file,
            add_reg_names=['Translation along x axis',
                           'Translation along yaxis',
                           'Translation along z axis',
                           'Rotation along x axis',
                           'Rotation along y axis',
                           'Rotation along z axis',
                           'Zoom along x axis',
                           'Zoom along y axis',
                           'Zoom along z axis',
                           'Shear along x axis',
                           'Shear along y axis',
                           'Shear along z axis']
            )

        paradigms.append(paradigm)
        design_matrices.append(design_matrix)

        # convert contrasts to dict
        contrasts = dict((contrast_id,
                          # append zeros to end of contrast to match design
                          np.hstack((contrast_value, np.zeros(len(
                                design_matrix.names) - len(contrast_value)))))

                         for contrast_id, contrast_value in zip(
                contrast_ids, contrast_values))

        # more interesting contrasts
        contrasts['RH-LH'] = contrasts['RH'] - contrasts['LH']
        contrasts['LH-RH'] = -contrasts['RH-LH']
        contrasts['RF-LF'] = contrasts['RF'] - contrasts['LF']
        contrasts['LF-RF'] = -contrasts['RF-LF']
        contrasts['H'] = contrasts['RH'] + contrasts['LH']
        contrasts['F'] = contrasts['RF'] + contrasts['LF']
        contrasts['H-F'] = contrasts['RH'] + contrasts['LH'] - (
            contrasts['RF'] - contrasts['LF'])
        contrasts['F-H'] = -contrasts['H-F']

    # smooth the data
    if np.sum(fwhm) > 0:
        print "Smoothing fMRI data (fwhm = %s)..." % fwhm
        fmri_files = do_subject_preproc(SubjectData(
                func=fmri_files, output_dir=subject_output_dir),
                                       do_realign=False,
                                       do_coreg=False,
                                       do_normalize=False,
                                       fwhm=fwhm,
                                       do_report=False).func
        print "... done.\r\n"

    # fit GLM
    print (
        'Fitting a "Fixed Effect" GLM for merging LR and RL phase-encoding '
        'directions for subject %s ...' % subject_id)
    fmri_glm = FMRILinearModel(fmri_files,
                               [design_matrix.matrix
                                for design_matrix in design_matrices],
                               mask='compute'
                               )
    fmri_glm.fit(do_scaling=True, model='ar1')
    print "... done.\r\n"

    # save computed mask
    mask_path = os.path.join(subject_output_dir, "mask.nii.gz")

    print "Saving mask image to %s ..." % mask_path
    nibabel.save(fmri_glm.mask, mask_path)
    print "... done.\r\n"

    # replicate contrasts across sessions
    contrasts = dict((cid, [cval] * 2)
                     for cid, cval in contrasts.iteritems())

    # compute effects
    for contrast_id, contrast_val in contrasts.iteritems():
        print "\tcontrast id: %s" % contrast_id
        z_map, t_map, eff_map, var_map = fmri_glm.contrast(
            contrast_val,
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True
            )

        # store stat maps to disk
        for map_type, out_map in zip(['z', 't', 'effects', 'variance'],
                                     [z_map, t_map, eff_map, var_map]):
            map_dir = os.path.join(
                subject_output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            print "\t\tWriting %s ..." % map_path
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if map_type == 'z':
                z_maps[contrast_id] = map_path

            if map_type == 'effects':
                effects_maps[contrast_id] = map_path

    # do stats report
    stats_report_filename = os.path.join(subject_output_dir,
                                         "report_stats.html")
    generate_subject_stats_report(
        stats_report_filename,
        dict((cid, cval[0][:len(fsl_condition_ids)])
             for cid, cval in contrasts.iteritems()),
        z_maps,
        fmri_glm.mask,
        threshold=3.,
        slicer='y',
        cut_coords=6,
        design_matrices=design_matrices,
        subject_id=subject_id,
        start_time=stats_start_time,
        title="GLM for subject %s" % subject_id,

        # additional ``kwargs`` for more informative report
        paradigm={'LR': paradigms[0].__dict__, 'RL': paradigms[0].__dict__},
        TR=tr,
        n_scans=n_scans,
        hfcut=hfcut,
        frametimes=frametimes,
        drift_model=drift_model,
        hrf_model=hrf_model,
        fwhm=fwhm
        )

    ProgressReport().finish_dir(subject_output_dir)
    print "\r\nStatistic report written to %s\r\n" % stats_report_filename

if __name__ == '__main__':
    subject_data_dir = "/home/elvis/CODE/datasets/100307"
    subject_output_dir = os.path.join(subject_data_dir, "nipy_results")
    run_suject_level1_glm(subject_data_dir, subject_output_dir, fwhm=4.)
