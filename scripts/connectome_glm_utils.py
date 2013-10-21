"""
:Author: DOHMATOB Elvis Dopgima

"""

import os
import glob
import numpy as np
import nibabel
import commands
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs.mask import intersect_masks
from pypreprocess.nipype_preproc_spm_utils import (SubjectData,
                                                       do_subject_preproc)
from pypreprocess.io_utils import (load_specific_vol)
from pypreprocess.fsl_to_nipy import (read_design_fsl_design_file,
                                      make_dmtx_from_timing_files,
                                      _insert_directory_in_file_name)
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.reporting.base_reporter import (ProgressReport,
                                                  pretty_time)
from joblib import Parallel, delayed, Memory


def run_suject_level1_glm(subject_data_dir, subject_output_dir, task_id,
                          readout_time=.01392,  # seconds
                          tr=.72,
                          do_dc=False,
                          do_realign=False,
                          do_coreg=False,
                          do_segment=False,
                          do_normalize=False,
                          fwhm=0.,
                          hrf_model="Canonical with Derivative",
                          drift_model="Cosine",
                          hfcut=100,
                          regress_motion=True,
                          slicer='z',
                          cut_coords=6,
                          threshold=3.,
                          cluster_th=15
                          ):

    # sanitize subject data_dir
    subject_id = int(os.path.basename(subject_data_dir))
    subject_data_dir = os.path.abspath(subject_data_dir)
    _subject_data_dir = os.path.join(subject_data_dir,
                                     "MNINonLinear/Results/")

    add_regs_files = None

    if do_dc or do_realign or do_coreg or do_normalize:
        subject_output_dir = subject_output_dir + "_custom_preproc"

        # glob fmri files
        fmri_files = [os.path.join(
                subject_data_dir,
                "/volatile/home/edohmato/datasets/100307/%s.nii.gz" % direction)
                      for direction in ["LR", "RL"]]
        
                # ""
                # "unprocessed/3T/tfMRI_%s_%s/%s_3T_tfMRI_%s_%s.nii.gz" % (
                #     task_id, direction, subject_id,
                #     task_id, direction))
                #           for direction in ["LR", "RL"]]
        assert len(fmri_files) == 2

        # glob anat file
        anat_file = os.path.join(subject_data_dir,
                                 "T1w/T1w_acpc_dc_restore_brain.nii.gz")
        assert os.path.isfile(anat_file)

        # prepare for smart caching
        mem = Memory(os.path.join(subject_output_dir, "cache_dir"))

        ###########################
        # distortion correction ?
        ###########################
        if do_dc:
            acq_params = [[-1, 0, 0, readout_time], [1, 0, 0, readout_time]]
            acq_params_file = os.path.join(subject_output_dir,
                                           "b0_acquisition_params.txt")
            np.savetxt(acq_params_file, acq_params, fmt='%f')

            fieldmap_files = [os.path.join(
                    subject_data_dir,
                    "unprocessed/3T/tfMRI_%s_%s/%s_3T_SpinEchoFieldMap_"
                    "%s.nii.gz" % (task_id, direction, subject_id, direction))
                              for direction in ["LR", "RL"]]
            assert len(fieldmap_files) == 2

            # fslroi
            zeroth_fieldmap_files = []
            for fieldmap_file in fieldmap_files:
                zeroth_fieldmap_file = os.path.join(
                    subject_output_dir, "0th_%s" % os.path.basename(
                        fieldmap_file))
                fslroi_cmd = "fsl5.0-fslroi %s %s 0 1" % (
                    fieldmap_file, zeroth_fieldmap_file)
                print "\r\nExecuting %s ..." % fslroi_cmd
                print mem.cache(commands.getoutput)(fslroi_cmd)

                zeroth_fieldmap_files.append(zeroth_fieldmap_file)

            # fslmerge
            merged_zeroth_fieldmap_file = os.path.join(
                subject_output_dir, "merged_with_other_direction_%s" % (
                    os.path.basename(zeroth_fieldmap_files[0])))
            fslmerge_cmd = "fslmerge -t %s %s %s" % (
                merged_zeroth_fieldmap_file, zeroth_fieldmap_files[0],
                zeroth_fieldmap_files[1])
            print "\r\nExecuting %s ..." % fslmerge_cmd
            print mem.cache(commands.getoutput)(fslmerge_cmd)

            # topup
            topup_results_basename = os.path.join(subject_output_dir,
                                                  "topup_results")
            topup_cmd = (
                "fsl5.0-topup --imain=%s --datain=%s --config=b02b0.cnf "
                "--out=%s" % (merged_zeroth_fieldmap_file, acq_params_file,
                              topup_results_basename))
            print "\r\nExecuting %s ..." % topup_cmd
            print mem.cache(commands.getoutput)(topup_cmd)

            # apply topup
            dc_fmri_files = []
            for index in xrange(2):
                dc_fmri_file = os.path.join(
                    subject_output_dir, "dc" + os.path.basename(
                        fmri_files[index]))
                applytopup_cmd = (
                    "fsl5.0-applytopup --imain=%s --verbose --inindex=%i "
                    "--topup=%s --out=%s --datain=%s --method=jac" % (
                        fmri_files[0], index + 1, topup_results_basename,
                        dc_fmri_file, acq_params_file))
                print "\r\nExecuting %s ..." % applytopup_cmd
                print mem.cache(commands.getoutput)(applytopup_cmd)

                dc_fmri_files.append(dc_fmri_file)

            fmri_files = dc_fmri_files

        # preprocess the data
        preproc_subject_data = do_subject_preproc(SubjectData(
                func=fmri_files, anat=anat_file,
                output_dir=subject_output_dir),
                                                  do_realign=do_realign,
                                                  do_coreg=do_coreg,
                                                  do_segment=do_segment,
                                                  do_normalize=do_normalize,
                                                  fwhm=fwhm,
                                                  do_report=False
                                                  )
        fmri_files = preproc_subject_data.func
        n_motion_regressions = 6
        if do_realign and regress_motion:
            add_regs_files = preproc_subject_data.realignment_parameters
    else:
        n_motion_regressions = 12

        # glob fmri files
        fmri_files = [os.path.join(
                _subject_data_dir, "tfMRI_%s_%s/tfMRI_%s_%s.nii.gz" % (
                    task_id, direction, task_id, direction))
                      for direction in ['LR', 'RL']]
        assert len(fmri_files) == 2

        # glob movement confounds
        if regress_motion:
            add_regs_files = [os.path.join(_subject_data_dir,
                                           "tfMRI_%s_%s" % (
                        task_id, direction),
                                           "Movement_Regressors.txt")
                              for direction in ["LR", "RL"]]

        # smooth images
        if np.sum(fwhm) > 0:
            print "Smoothing fMRI data (fwhm = %s)..." % fwhm
            fmri_files = do_subject_preproc(SubjectData(
                    func=fmri_files, output_dir=subject_output_dir),
                                            do_realign=False,
                                            do_coreg=False,
                                            do_segment=True,
                                            do_normalize=False,
                                            fwhm=fwhm,
                                            do_report=False
                                            ).func
            print "... done.\r\n"

    # sanitize subject_output_dir
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)

    # chronometry
    stats_start_time = pretty_time()

    # merged lists
    paradigms = []
    frametimes_list = []
    design_matrices = []
    # fmri_files = []
    n_scans = []
    for direction, direction_index in zip(['LR', 'RL'], xrange(2)):
        # glob the design file
        design_file = os.path.join(_subject_data_dir, "tfMRI_%s_%s" % (
                task_id, direction),
                                   "tfMRI_%s_%s_hp200_s4_level1.fsf" % (
                task_id, direction))

        # read the experimental setup
        print "Reading experimental setup from %s ..." % design_file
        fsl_condition_ids, timing_files, fsl_contrast_ids, contrast_values = \
            read_design_fsl_design_file(design_file)
        print "... done.\r\n"

        # fix timing filenames
        timing_files = _insert_directory_in_file_name(
            timing_files, "tfMRI_%s_%s" % (task_id, direction), 1)

        # make design matrix
        print "Constructing design matrix for direction %s ..." % direction
        _n_scans = nibabel.load(fmri_files[direction_index]).shape[-1]
        n_scans.append(_n_scans)
        design_matrix, paradigm, frametimes = make_dmtx_from_timing_files(
            timing_files, fsl_condition_ids, n_scans=_n_scans, tr=tr,
            hrf_model=hrf_model, drift_model=drift_model, hfcut=hfcut,
            add_regs_file=add_regs_files[
                direction_index] if not add_regs_files is None else None,
            add_reg_names=[
                'Translation along x axis',
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
                'Shear along z axis'][:n_motion_regressions
                                       ] if not add_regs_files is None
            else None,
            )
        print "... done."

        paradigms.append(paradigm)
        frametimes_list.append(frametimes)
        design_matrices.append(design_matrix)

        # convert contrasts to dict
        contrasts = dict((contrast_id,
                          # append zeros to end of contrast to match design
                          np.hstack((contrast_value, np.zeros(len(
                                design_matrix.names) - len(contrast_value)))))

                         for contrast_id, contrast_value in zip(
                fsl_contrast_ids, contrast_values))

        # more interesting contrasts
        if task_id == 'MOTOR':
            contrasts['RH-LH'] = contrasts['RH'] - contrasts['LH']
            contrasts['LH-RH'] = -contrasts['RH-LH']
            contrasts['RF-LF'] = contrasts['RF'] - contrasts['LF']
            contrasts['LF-RF'] = -contrasts['RF-LF']
            contrasts['H'] = contrasts['RH'] + contrasts['LH']
            contrasts['F'] = contrasts['RF'] + contrasts['LF']
            contrasts['H-F'] = contrasts['RH'] + contrasts['LH'] - (
                contrasts['RF'] - contrasts['LF'])
            contrasts['F-H'] = -contrasts['H-F']

    # importat maps
    z_maps = {}
    effects_maps = {}

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

    # remove repeated contrasts
    contrasts = dict((cid, cval[0]) for cid, cval in contrasts.iteritems())

    # do stats report
    anat_img = load_specific_vol(fmri_files[0], 0)[0]
    stats_report_filename = os.path.join(subject_output_dir,
                                         "report_stats.html")
    generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        z_maps,
        fmri_glm.mask,
        anat=anat_img.get_data(),
        anat_affine=anat_img.get_affine(),
        threshold=threshold,
        cluster_th=cluster_th,
        slicer=slicer,
        cut_coords=cut_coords,
        design_matrices=design_matrices,
        subject_id=subject_id,
        start_time=stats_start_time,
        title="GLM for subject %s" % subject_id,

        # additional ``kwargs`` for more informative report
        TR=tr,
        n_scans=n_scans,
        hfcut=hfcut,
        drift_model=drift_model,
        hrf_model=hrf_model,
        paradigm={'LR': paradigms[0].__dict__, 'RL': paradigms[1].__dict__},
        frametimes={'LR': frametimes_list[0], 'RL': frametimes_list[1]},
        fwhm=fwhm
        )

    ProgressReport().finish_dir(subject_output_dir)
    print "\r\nStatistic report written to %s\r\n" % stats_report_filename

    return contrasts, effects_maps, z_maps, mask_path

if __name__ == '__main__':
    data_dir = "/media/HCP-Q1-Reproc/"
    output_dir = "/mnt/3t/edohmato/connectome_output/hcp_preproc"

    n_jobs = int(os.environ['N_JOBS']) if 'N_JOBS' in os.environ else -1
    n_subjects = int(os.environ['N_SUBJECTS']
                     ) if 'N_SUBJECTS' in os.environ else -1

    def _run_suject_level1_glm(subject_data_dir, subject_output_dir,
                              **kwargs):
        mem = Memory(os.path.join(subject_output_dir, "cache_dir"))
        return mem.cache(run_suject_level1_glm)(subject_data_dir,
                                                subject_output_dir,
                                                **kwargs)

    def _subject_factory(task_output_dir, n_subjects=-1):
        """
        Generator for subject data.

        Returns
        -------
        subject_data_dir: string
            existing directory; directory containing subject data
        subject_output_dir: string
            output directory for subject GLM

        """

        for subject_data_dir in sorted(glob.glob(os.path.join(
                    data_dir, "****"))):
            if n_subjects == 0:
                return
            else:
                n_subjects -= 1

            if not os.path.isdir(subject_data_dir):
                continue

            subject_id = os.path.basename(subject_data_dir)
            if subject_id != "100307":
                continue
            subject_output_dir = os.path.join(task_output_dir, subject_id)

            yield subject_data_dir, subject_output_dir

    # run batch single-subject (level 1) GLM analysis on all subjects
    # and collect the results; this will be input for level 2 analysis
    slicer = 'z'
    cut_coords = 5
    threshold = 2.3
    cluster_th = 15
    for task_id in [
        "MOTOR",
        #'LANGUAGE',
        #"EMOTION"
        # "WM"
        ]:
        # best slicer for given task
        if task_id == "MOTOR":
            slicer = 'y'

        task_output_dir = os.path.join(output_dir, task_id)

        # chronometry
        stats_start_time = pretty_time()

        level1_results = Parallel(n_jobs=n_jobs, verbose=100)(
            delayed(_run_suject_level1_glm)(
                subject_data_dir,
                subject_output_dir,
                task_id=task_id,
                do_dc=False,  # True,
                do_realign=False,  # True,
                do_coreg=True,
                do_segment=True,
                do_normalize=True,
                regress_motion=True,
                fwhm=4.,
                slicer=slicer,
                cut_coords=cut_coords,
                threshold=threshold,
                cluster_th=cluster_th
                )
            for subject_data_dir, subject_output_dir in _subject_factory(
                task_output_dir, n_subjects=n_subjects))

        # compute group mask
        print "\r\nComputing group mask ..."
        mask_images = [level1_result[3] for level1_result in level1_results]
        grp_mask = nibabel.Nifti1Image(intersect_masks(mask_images
                                                       ).astype(np.uint8),
                                       nibabel.load(mask_images[0]
                                                    ).get_affine())
        print "... done.\r\n"

        print "Group GLM"
        level1_contrasts = [level1_result[0]
                            for level1_result in level1_results]
        level1_contrasts = level1_results[0][0]
        level1_effects_maps = [level1_result[1]
                               for level1_result in level1_results]
        second_level_z_maps = {}
        design_matrix = np.ones(len(level1_effects_maps)
                                )[:, np.newaxis]  # only the intercept
        for contrast_id in level1_contrasts:
            print "\tcontrast id: %s" % contrast_id

            # effects maps will be the input to the second level GLM
            first_level_image = nibabel.concat_images(
                [x[contrast_id] for x in level1_effects_maps])

            # fit 2nd level GLM for given contrast
            grp_model = FMRILinearModel(first_level_image,
                                        design_matrix, grp_mask)
            grp_model.fit(do_scaling=False, model='ols')

            # specify and estimate the contrast
            contrast_val = np.array(([[1.]]))  # the only possible contrast !
            z_map, = grp_model.contrast(contrast_val,
                                        con_id='one_sample %s' % contrast_id,
                                        output_z=True)

            # save map
            map_dir = os.path.join(task_output_dir, 'z_maps')
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, '2nd_level_%s.nii.gz' % (
                    contrast_id))
            print "\t\tWriting %s ..." % map_path
            nibabel.save(z_map, map_path)

            second_level_z_maps[contrast_id] = map_path

        # do stats report
        stats_report_filename = os.path.join(task_output_dir,
                                             "report_stats.html")
        generate_subject_stats_report(
            stats_report_filename,
            level1_contrasts,
            second_level_z_maps,
            grp_mask,
            threshold=threshold,
            cluster_th=cluster_th,
            design_matrices=[design_matrix],
            subject_id="sub001",
            start_time=stats_start_time,
            title='Group GLM for HCP fMRI %s task' % task_id,
            slicer=slicer,
            cut_coords=cut_coords
            )

        ProgressReport().finish_dir(task_output_dir)
        print "\r\nStatistic report written to %s\r\n" % stats_report_filename
