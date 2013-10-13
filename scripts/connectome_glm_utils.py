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


def run_suject_level1_glm(data_dir, direction, tr=.72,
                          task="MOTOR",
                          fwhm=4.,
                          output_dir=None,
                          hrf_model="Canonical with Derivative",
                          drift_model="Cosine",
                          hfcut=128,
                          regress_motion=True,
                          motion_params_file_pattern="Movement_Regressors.txt"
                          ):
    # output dir
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'nipy_results', direction)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # subject_id
    subject_id = os.path.basename(data_dir)

    # glob the design file
    design_file = os.path.join(data_dir, "tfMRI_%s_%s" % (task, direction),
                               "tfMRI_%s_%s_hp200_s4_level1.fsf" % (
            task, direction))

    # glog motion parameter files
    add_regs_file = None
    if regress_motion:
        add_regs_file = os.path.join(data_dir,
                                      "tfMRI_%s_%s" % (task, direction),
                                      motion_params_file_pattern)

    # glob the preprocessed 4D fmri file
    fmri_file = os.path.join(data_dir, "tfMRI_%s_%s" % (task, direction),
                             "tfMRI_MOTOR_%s.nii.gz" % direction)

    # chronometry
    stats_start_time = pretty_time()

    # read the experimental setup
    print "Reading experimental setup from %s ..." % design_file
    experiment = read_design_fsl_design_file(design_file)
    fsl_condition_ids, timing_files, contrast_ids, contrast_values = experiment
    print "... done.\r\n"

    # fix timing filenames
    timing_files = _insert_directory_in_file_name(
        timing_files, "tfMRI_%s_%s" % (task, direction), 1)

    # make design matrix
    n_scans = nibabel.load(fmri_file).shape[-1]
    design_matrix, paradigm, frametimes = make_dmtx_from_timing_files(
        timing_files, fsl_condition_ids, n_scans=n_scans, tr=tr,
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

    # convert contrasts to dict
    contrasts = dict((contrast_id,
                      # append zeros to end of contrast to match design
                      np.hstack((contrast_value, np.zeros(len(
                            design_matrix.names) - len(contrast_value)))))

                     for contrast_id, contrast_value in zip(
            contrast_ids, contrast_values))

    # smooth the data
    if np.sum(fwhm) > 0:
        print "Smoothing fMRI data (fwhm = %s)..." % fwhm
        subject_data = SubjectData(func=fmri_file, output_dir=output_dir)
        fmri_file = do_subject_preproc(subject_data,
                                       do_realign=False,
                                       do_coreg=False,
                                       do_normalize=False,
                                       fwhm=fwhm,
                                       do_report=False).func
        print "... done.\r\n"

    # fit GLM
    print 'Fitting a GLM for %s (this takes time)...' % direction
    fmri_glm = FMRILinearModel(fmri_file,
                               design_matrix.matrix,
                               mask='compute'
                               )
    fmri_glm.fit(do_scaling=True, model='ar1')
    print "... done.\r\n"

    # save computed mask
    mask_path = os.path.join(output_dir, "mask.nii.gz")

    print "Saving mask image to %s ..." % mask_path
    nibabel.save(fmri_glm.mask, mask_path)
    print "... done.\r\n"

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
                output_dir, '%s_maps' % map_type)
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
    stats_report_filename = os.path.join(output_dir, "report_stats_%s.html" % (
            direction))
    generate_subject_stats_report(
        stats_report_filename,
        dict((cid, cval[:len(fsl_condition_ids)])
             for cid, cval in contrasts.iteritems()),
        z_maps,
        fmri_glm.mask,
        threshold=2.3,
        slicer='y',
        design_matrices=design_matrix,
        subject_id=subject_id,
        start_time=stats_start_time,
        title="GLM for subject %s, %s phase-encding direction" % (subject_id,
                                                                  direction
                                                                  ),

        # additional ``kwargs`` for more informative report
        paradigm=paradigm.__dict__,
        TR=tr,
        n_scans=n_scans,
        hfcut=hfcut,
        frametimes=frametimes,
        drift_model=drift_model,
        hrf_model=hrf_model,
        )

    ProgressReport().finish_dir(output_dir)
    print "Statistic report written to %s\r\n" % stats_report_filename

if __name__ == '__main__':
    run_suject_level1_glm("/home/elvis/CODE/datasets/100307", "RL")
