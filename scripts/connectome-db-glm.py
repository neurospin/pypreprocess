"""
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

import os
import glob
import numpy as np
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs.mask import intersect_masks
import nibabel
import time
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.reporting.base_reporter import ProgressReport
# from pypreprocess.io_utils import compute_mean_3D_image
from pypreprocess.nipype_preproc_spm_utils_bis import (do_subject_preproc,
                                                       SubjectData
                                                       )

TR = 720. / 1000  # XXX ???

# paths
OUTPUT_DIR = "/home/elvis/CODE/datasets/connectome-glm"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def run_subject_glm(subject_id, fmri_files, me_files, mr_files, fwhm=0.):
    subject_data = SubjectData(func=fmri_files,
                               subject_id=subject_id,
                               output_dir=os.path.join(
            OUTPUT_DIR, subject_id)
                               )

    # preprocess the data
    if np.sum(fwhm) > 0:
        subject_data = do_subject_preproc(subject_data,
                                          do_realign=False,
                                          do_coreg=False,
                                          do_normalize=False,
                                          fwhm=[8, 8, 8],
                                          do_report=False)

    # build experimental paradigm
    stats_start_time = time.ctime()
    design_matrices = []
    first_level_effects_maps = []
    mask_images = []
    for direc in ["LR", "RL"]:
        index = ["LR", "RL"].index(direc)
        n_scans = nibabel.load(subject_data.func[index]).shape[-1]

        subject_session_output_dir = os.path.join(subject_data.output_dir,
                                                  "session_stats",
                                                  direc)
        if not os.path.exists(subject_session_output_dir):
            os.makedirs(subject_session_output_dir)

        conditions = []
        onsets = []
        durations = []
        amplitudes = []
        for ev_file in ev_files:
            if not direc in ev_file:
                continue

            try:
                timing = np.loadtxt(ev_file)
                if timing.ndim > 1:
                    condition_name = os.path.basename(ev_file).lower(
                        ).split('.')[0]
                    conditions = conditions + [condition_name
                                               ] * timing.shape[0]
                    onsets = onsets + list(timing[..., 0])
                    durations = durations + list(timing[..., 1])
                    amplitudes = amplitudes + list(timing[..., 2])
            except (OSError, IOError, TypeError, ValueError):
                continue

        paradigm = BlockParadigm(con_id=conditions, onset=onsets,
                                 duration=durations, amplitude=amplitudes)

        # build design matrix
        frametimes = np.linspace(0, (n_scans - 1) * TR, n_scans)
        hfcut = 128
        drift_model = 'Cosine'
        hrf_model = 'Canonical With Derivative'
        design_matrix = make_dmtx(frametimes, paradigm, hrf_model=hrf_model,
                                  drift_model=drift_model, hfcut=hfcut,
                                  # add_reg_names=['MR%i' % (i + 1)
                                  #                for i in xrange(12)],
                                  # add_regs=np.loadtxt(mr_files[index])
                                  )
        design_matrices.append(design_matrix)

        # specify contrasts
        contrasts = {}
        I = np.eye(len(design_matrix.names))
        for i in xrange(paradigm.n_conditions):
            contrasts[design_matrix.names[2 * i]] = I[2 * i]

        # more interesting contrasts
        contrasts["effects_of_interest"] = np.array(contrasts.values()
                                                    ).sum(axis=0)
        contrasts["lh-rh"] = contrasts['lh'] - contrasts['rh']
        contrasts["rh-lh"] = -contrasts["lh-rh"]
        contrasts["lf-rf"] = contrasts['lf'] - contrasts['rf']
        contrasts["rf-lf"] = -contrasts["lf-rf"]
        contrasts["hands-feet"] = contrasts['lh'] + contrasts[
            'rh'] - contrasts['lf'] - contrasts['rf']
        contrasts["feet-hands"] = -contrasts["hands-feet"]

        # fit GLM
        print 'Fitting a GLM for %s (this takes time)...' % (
            direc)
        fmri_glm = FMRILinearModel(subject_data.func[index],
                                   design_matrix.matrix,
                                   mask='compute'
                                   )
        fmri_glm.fit(do_scaling=True, model='ar1')

        # save computed mask
        mask_path = os.path.join(subject_session_output_dir,
                                 "mask.nii.gz")
        print "Saving mask image %s" % mask_path
        nibabel.save(fmri_glm.mask, mask_path)
        mask_images.append(mask_path)

        # compute contrasts
        z_maps = {}
        effects_maps = {}
        for contrast_id, contrast_val in contrasts.iteritems():
            print "\tcontrast id: %s" % contrast_id
            z_map, t_map, effects_map, var_map = fmri_glm.contrast(
                contrast_val,
                con_id=contrast_id,
                output_z=True,
                output_stat=True,
                output_effects=True,
                output_variance=True
                )

            # store stat maps to disk
            for map_type, out_map in zip(['z', 't', 'effects', 'variance'],
                                      [z_map, t_map, effects_map, var_map]):
                map_dir = os.path.join(
                    subject_session_output_dir, '%s_maps' % map_type)
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

        first_level_effects_maps.append(effects_maps)

        # do stats report
        stats_report_filename = os.path.join(subject_session_output_dir,
                                             "report_stats_%s.html" % direc)

        generate_subject_stats_report(
            stats_report_filename,
            contrasts,
            z_maps,
            fmri_glm.mask,
            threshold=3.,
            slicer='y',
            cut_coords=20,
            design_matrices=design_matrix,
            subject_id=subject_data.subject_id,
            start_time=stats_start_time,
            title="GLM for subject %s, session %s" % (subject_data.subject_id,
                                                      direc
                                                      ),

            # additional ``kwargs`` for more informative report
            paradigm=paradigm.__dict__,
            TR=TR,
            n_scans=n_scans,
            hfcut=hfcut,
            frametimes=frametimes,
            drift_model=drift_model,
            hrf_model=hrf_model,
            )

        ProgressReport().finish_dir(subject_session_output_dir)
        print "Statistic report written to %s\r\n" % stats_report_filename

    # compute a population-level mask as the intersection of individual masks
    print "Inter-session GLM"
    grp_mask = nibabel.Nifti1Image(intersect_masks(
            mask_images).astype(np.int8),
                                   nibabel.load(mask_images[0]).get_affine())
    second_level_z_maps = {}
    design_matrix = np.ones(len(first_level_effects_maps)
                            )[:, np.newaxis]  # only the intercept
    for contrast_id in contrasts:
        print "\tcontrast id: %s" % contrast_id

        # effects maps will be the input to the second level GLM
        first_level_image = nibabel.concat_images(
            [x[contrast_id] for x in first_level_effects_maps])

        # fit 2nd level GLM for given contrast
        grp_model = FMRILinearModel(first_level_image, design_matrix, grp_mask)
        grp_model.fit(do_scaling=False, model='ols')

        # specify and estimate the contrast
        contrast_val = np.array(([[1.]]))  # the only possible contrast !
        z_map, = grp_model.contrast(contrast_val,
                                    con_id='one_sample %s' % contrast_id,
                                    output_z=True)

        # save map
        map_dir = os.path.join(subject_data.output_dir, 'z_maps')
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(map_dir, '2nd_level_%s.nii.gz' % contrast_id)
        print "\t\tWriting %s ..." % map_path
        nibabel.save(z_map, map_path)

        second_level_z_maps[contrast_id] = map_path

    # do stats report
    stats_report_filename = os.path.join(subject_data.output_dir,
                                         "report_stats.html")
    generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        second_level_z_maps,
        grp_mask,
        threshold=2.,
        cluster_th=1,
        design_matrices=[design_matrix],
        subject_id="sub001",
        start_time=stats_start_time,
        title="Inter-session GLM for subject %s" % subject_data.subject_id,
        slicer='y',
        cut_coords=8
        )

    ProgressReport().finish_dir(subject_data.output_dir)
    print "\r\nStatistic report written to %s\r\n" % stats_report_filename

if __name__ == '__main__':
    DATA_DIR = "/home/elvis/CODE/datasets"

    # fetch the data
    subject_id = "100307"
    fmri_files = [os.path.join(DATA_DIR, subject_id,
                               'tfMRI_MOTOR_%s/tfMRI_MOTOR_%s.nii.gz' % (
                direc, direc)) for direc in ["LR", "RL"]]
    # EV files
    ev_files = sorted(glob.glob(os.path.join(DATA_DIR, subject_id,
                                             "tfMRI_MOTOR_*/EVs/*.txt")))

    # movement regressors
    mr_files = sorted(glob.glob(os.path.join(
                DATA_DIR, subject_id,
                "tfMRI_MOTOR_*/Movement_Regressors.txt")))

    run_subject_glm(subject_id, fmri_files, ev_files, mr_files)

    run_subject_glm("100307",
                    fmri_files=['/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/100307_3T_tfMRI_MOTOR_LR.nii.gz',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/100307_3T_tfMRI_MOTOR_RL.nii.gz'],
                    ev_files='/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/LINKED_DATA/EPRIME/EVs/Sync.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/LINKED_DATA/EPRIME/EVs/cue.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/LINKED_DATA/EPRIME/EVs/lf.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/LINKED_DATA/EPRIME/EVs/lh.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/LINKED_DATA/EPRIME/EVs/rf.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/LINKED_DATA/EPRIME/EVs/rh.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR/LINKED_DATA/EPRIME/EVs/t.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_RL/LINKED_DATA/EPRIME/EVs/Sync.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_RL/LINKED_DATA/EPRIME/EVs/cue.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_RL/LINKED_DATA/EPRIME/EVs/lf.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_RL/LINKED_DATA/EPRIME/EVs/lh.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_RL/LINKED_DATA/EPRIME/EVs/rf.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_RL/LINKED_DATA/EPRIME/EVs/rh.txt',
 '/home/elvis/CODE/datasets/connectome-db/100307/tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_RL/LINKED_DATA/EPRIME/EVs/t.txt'],
    
