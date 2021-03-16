"""
:Synopsis: preprocessing and/or analysis of HCP task fMRI data
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

# bulletproof: we don't need X & co.
import matplotlib
matplotlib.use('Agg')

import os
import sys
import numpy as np
import nibabel
import subprocess
from nipy.modalities.fmri.glm import FMRILinearModel
from nilearn.masking import intersect_masks
from pypreprocess.nipype_preproc_spm_utils import (SubjectData,
                                                   _do_subject_realign,
                                                   do_subject_preproc)
from pypreprocess.io_utils import load_specific_vol
from pypreprocess.fsl_to_nipy import (read_fsl_design_file,
                                      make_dmtx_from_timing_files)
from pypreprocess.reporting.pypreproc_reporter import pretty_time
from pypreprocess.conf_parser import _generate_preproc_pipeline
from joblib import Parallel, delayed, Memory
from nipype.caching import Memory as NipypeMemory
import nipype.interfaces.spm as spm
from pypreprocess.io_utils import hard_link


def _do_fmri_distortion_correction(subject_data,
                                   # i'm unsure of the readout time,
                                   # but this is constant across both PE
                                   # directions and so can be scaled to 1
                                   # (or any other nonzero float)
                                   protocol="MOTOR",
                                   readout_time=.01392,
                                   realign=True,
                                   coregister=True,
                                   coreg_func_to_anat=True,
                                   dc=True,
                                   segment=False,
                                   normalize=False,
                                   func_write_voxel_sizes=None,
                                   anat_write_voxel_sizes=None,
                                   report=False,
                                   **kwargs
                                   ):
    """
    Function to undistort task fMRI data for a given HCP subject.

    """

    directions = ['LR', 'RL']

    subject_data.sanitize()

    if dc:
        acq_params = [[1, 0, 0, readout_time], [-1, 0, 0, readout_time]]
        acq_params_file = os.path.join(subject_data.output_dir,
                                       "b0_acquisition_params.txt")
        np.savetxt(acq_params_file, acq_params, fmt='%f')

        fieldmap_files = [os.path.join(os.path.dirname(
                    subject_data.func[sess]),
                                       "%s_3T_SpinEchoFieldMap_%s.nii.gz" % (
                    subject_data.subject_id, directions[sess]))
                          for sess in range(subject_data.n_sessions)]
        sbref_files = [sess_func.replace(".nii", "_SBRef.nii")
                       for sess_func in subject_data.func]

        # prepare for smart caching
        mem = Memory(os.path.join(subject_data.output_dir, "cache_dir"))

        for x in [fieldmap_files, sbref_files, subject_data.func]:
            assert len(x) == 2
            for y in x:
                assert os.path.isfile(y), y

        # fslroi
        zeroth_fieldmap_files = []
        for fieldmap_file in fieldmap_files:
            if not os.path.isfile(fieldmap_file):
                print("Can't find fieldmap file %s; skipping subject %s" %
                      fieldmap_file, subject_data.subject_id)
                return

            # peel 0th volume of each fieldmap
            zeroth_fieldmap_file = os.path.join(
                subject_data.output_dir, "0th_%s" % os.path.basename(
                    fieldmap_file))
            fslroi_cmd = "fsl5.0-fslroi %s %s 0 1" % (
                fieldmap_file, zeroth_fieldmap_file)
            print("\r\nExecuting '%s' ..." % fslroi_cmd)
            print(mem.cache(subprocess.check_output)(fslroi_cmd))

            zeroth_fieldmap_files.append(zeroth_fieldmap_file)

        # merge the 0th volume of both fieldmaps
        merged_zeroth_fieldmap_file = os.path.join(
            subject_data.output_dir, "merged_with_other_direction_%s" % (
                os.path.basename(zeroth_fieldmap_files[0])))
        fslmerge_cmd = "fsl5.0-fslmerge -t %s %s %s" % (
            merged_zeroth_fieldmap_file, zeroth_fieldmap_files[0],
            zeroth_fieldmap_files[1])
        print("\r\nExecuting '%s' ..." % fslmerge_cmd)
        print(mem.cache(subprocess.check_output)(fslmerge_cmd))

        # do topup (learn distortion model)
        topup_results_basename = os.path.join(subject_data.output_dir,
                                              "topup_results")
        topup_cmd = (
            "fsl5.0-topup --imain=%s --datain=%s --config=b02b0.cnf "
            "--out=%s" % (merged_zeroth_fieldmap_file, acq_params_file,
                          topup_results_basename))
        print("\r\nExecuting '%s' ..." % topup_cmd)
        print(mem.cache(subprocess.check_output)(topup_cmd))

        # apply learn deformations to absorb distortion
        dc_fmri_files = []

        for sess in range(2):
            # merge SBRef + task BOLD for current PE direction
            assert len(subject_data.func) == 2, subject_data
            fourD_plus_sbref = os.path.join(
                subject_data.output_dir, "sbref_plus_" + os.path.basename(
                    subject_data.func[sess]))
            fslmerge_cmd = "fsl5.0-fslmerge -t %s %s %s" % (
                fourD_plus_sbref, sbref_files[sess], subject_data.func[sess])
            print("\r\nExecuting '%s' ..." % fslmerge_cmd)
            print(mem.cache(subprocess.check_output)(fslmerge_cmd))

            # realign task BOLD to SBRef
            sess_output_dir = subject_data.session_output_dirs[sess]
            rfourD_plus_sbref = _do_subject_realign(SubjectData(
                    func=[fourD_plus_sbref],
                    output_dir=subject_data.output_dir,
                    n_sessions=1, session_output_dirs=[sess_output_dir]),
                                                    report=False).func[0]

            # apply topup to realigned images
            dc_rfourD_plus_sbref = os.path.join(
                subject_data.output_dir, "dc" + os.path.basename(
                    rfourD_plus_sbref))
            applytopup_cmd = (
                "fsl5.0-applytopup --imain=%s --verbose --inindex=%i "
                "--topup=%s --out=%s --datain=%s --method=jac" % (
                    rfourD_plus_sbref, sess + 1, topup_results_basename,
                    dc_rfourD_plus_sbref, acq_params_file))
            print("\r\nExecuting '%s' ..." % applytopup_cmd)
            print(mem.cache(subprocess.check_output)(applytopup_cmd))

            # recover undistorted task BOLD
            dc_rfmri_file = dc_rfourD_plus_sbref.replace("sbref_plus_", "")
            fslroi_cmd = "fsl5.0-fslroi %s %s 1 -1" % (
                dc_rfourD_plus_sbref, dc_rfmri_file)
            print("\r\nExecuting '%s' ..." % fslroi_cmd)
            print(mem.cache(subprocess.check_output)(fslroi_cmd))

            # sanity tricks
            if dc_rfmri_file.endswith(".nii"):
                dc_rfmri_file = dc_rfmri_file + ".gz"

            dc_fmri_files.append(dc_rfmri_file)

        subject_data.func = dc_fmri_files
        if isinstance(subject_data.func, str):
            subject_data.func = [subject_data.func]

    # continue preprocessing
    subject_data = do_subject_preproc(
        subject_data,
        realign=realign,
        coregister=coregister,
        coreg_anat_to_func=not coreg_func_to_anat,
        segment=True,
        normalize=False,
        report=report)

    # ok for GLM now
    return subject_data


def run_suject_level1_glm(subject_data,
                          readout_time=.01392,  # seconds
                          tr=.72,
                          dc=True,
                          hrf_model="spm + derivative",
                          drift_model="Cosine",
                          hfcut=100,
                          regress_motion=True,
                          slicer='ortho',
                          cut_coords=None,
                          threshold=3.,
                          cluster_th=15,
                          normalize=True,
                          fwhm=0.,
                          protocol="MOTOR",
                          func_write_voxel_sizes=None,
                          anat_write_voxel_sizes=None,
                          **other_preproc_kwargs
                          ):
    """
    Function to do preproc + analysis for a single HCP subject (task fMRI)

    """

    add_regs_files = None
    n_motion_regressions = 6
    subject_data.n_sessions = 2

    subject_data.tmp_output_dir = os.path.join(subject_data.output_dir, "tmp")
    if not os.path.exists(subject_data.tmp_output_dir):
        os.makedirs(subject_data.tmp_output_dir)

    if not os.path.exists(subject_data.output_dir):
        os.makedirs(subject_data.output_dir)

    mem = Memory(os.path.join(subject_data.output_dir, "cache_dir"),
                 verbose=100)

    # glob design files (.fsf)
    subject_data.design_files = [os.path.join(
            subject_data.data_dir, ("MNINonLinear/Results/tfMRI_%s_%s/"
                                    "tfMRI_%s_%s_hp200_s4_level1.fsf") % (
                protocol, direction, protocol, direction))
            for direction in ['LR', 'RL']]

    assert len(subject_data.design_files) == 2
    for df in subject_data.design_files:
        if not os.path.isfile(df):
            return

    if 0x0:
        subject_data = _do_fmri_distortion_correction(
            subject_data, dc=dc, fwhm=fwhm, readout_time=readout_time,
            **other_preproc_kwargs)

    # chronometry
    stats_start_time = pretty_time()

    # merged lists
    paradigms = []
    frametimes_list = []
    design_matrices = []
    # fmri_files = []
    n_scans = []
    # for direction, direction_index in zip(['LR', 'RL'], range(2)):
    for sess in range(subject_data.n_sessions):
        direction = ['LR', 'RL'][sess]
        # glob the design file
        # design_file = os.path.join(# _subject_data_dir, "tfMRI_%s_%s" % (
                # protocol, direction),
        design_file = subject_data.design_files[sess]
                #                    "tfMRI_%s_%s_hp200_s4_level1.fsf" % (
                # protocol, direction))
        if not os.path.isfile(design_file):
            print("Can't find design file %s; skipping subject %s" %
                  design_file, subject_data.subject_id)
            return

        # read the experimental setup
        print("Reading experimental setup from %s ..." % design_file)
        fsl_condition_ids, timing_files, fsl_contrast_ids, contrast_values = \
            read_fsl_design_file(design_file)
        print("... done.\r\n")

        # fix timing filenames
        timing_files = [tf.replace("EVs", "tfMRI_%s_%s/EVs" % (
                    protocol, direction)) for tf in timing_files]

        # make design matrix
        print("Constructing design matrix for direction %s ..." % direction)
        _n_scans = nibabel.load(subject_data.func[sess]).shape[-1]
        n_scans.append(_n_scans)
        add_regs_file = add_regs_files[
            sess] if not add_regs_files is None else None
        design_matrix, paradigm, frametimes = make_dmtx_from_timing_files(
            timing_files, fsl_condition_ids, n_scans=_n_scans, tr=tr,
            hrf_model=hrf_model, drift_model=drift_model, hfcut=hfcut,
            add_regs_file=add_regs_file,
            add_reg_names=[
                'Translation along x axis',
                'Translation along yaxis',
                'Translation along z axis',
                'Rotation along x axis',
                'Rotation along y axis',
                'Rotation along z axis',
                'Differential Translation along x axis',
                'Differential Translation along yaxis',
                'Differential Translation along z axis',
                'Differential Rotation along x axis',
                'Differential Rotation along y axis',
                'Differential Rotation along z axis'
                ][:n_motion_regressions] if not add_regs_files is None
            else None,
            )

        print("... done.")
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
        if protocol == 'MOTOR':
            contrasts['RH-LH'] = contrasts['RH'] - contrasts['LH']
            contrasts['LH-RH'] = -contrasts['RH-LH']
            contrasts['RF-LF'] = contrasts['RF'] - contrasts['LF']
            contrasts['LF-RF'] = -contrasts['RF-LF']
            contrasts['H'] = contrasts['RH'] + contrasts['LH']
            contrasts['F'] = contrasts['RF'] + contrasts['LF']
            contrasts['H-F'] = contrasts['RH'] + contrasts['LH'] - (
                contrasts['RF'] - contrasts['LF'])
            contrasts['F-H'] = -contrasts['H-F']

        contrasts = dict((k, v) for k, v in contrasts.items() if "-" in k)

    # replicate contrasts across sessions
    contrasts = dict((cid, [cval] * 2)
                     for cid, cval in contrasts.items())

    cache_dir = cache_dir = os.path.join(subject_data.output_dir,
                                         'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    nipype_mem = NipypeMemory(base_dir=cache_dir)

    if 0x0:
        if np.sum(fwhm) > 0.:
            subject_data.func = nipype_mem.cache(spm.Smooth)(
                in_files=subject_data.func,
                fwhm=fwhm,
                ignore_exception=False,
                ).outputs.smoothed_files

    # fit GLM
    def tortoise(*args):
        print(args)
        print(
            'Fitting a "Fixed Effect" GLM for merging LR and RL '
            'phase-encoding directions for subject %s ...' %
            subject_data.subject_id)
        fmri_glm = FMRILinearModel(subject_data.func,
                                   [design_matrix.matrix
                                    for design_matrix in design_matrices],
                                   mask='compute'
                                   )
        fmri_glm.fit(do_scaling=True, model='ar1')
        print("... done.\r\n")

        # save computed mask
        mask_path = os.path.join(subject_data.output_dir, "mask.nii")
        print("Saving mask image to %s ..." % mask_path)
        nibabel.save(fmri_glm.mask, mask_path)
        print("... done.\r\n")

        z_maps = {}
        effects_maps = {}
        map_dirs = {}
        try:
            for contrast_id, contrast_val in contrasts.items():
                print("\tcontrast id: %s" % contrast_id)
                z_map, eff_map = fmri_glm.contrast(
                    contrast_val,
                    con_id=contrast_id,
                    output_z=True,
                    output_effects=True
                    )

                # store stat maps to disk
                for map_type, out_map in zip(['z', 'effects'],
                                             [z_map, eff_map]):
                    map_dir = os.path.join(
                        subject_data.output_dir, '%s_maps' % map_type)
                    map_dirs[map_type] = map_dir
                    if not os.path.exists(map_dir):
                        os.makedirs(map_dir)
                    map_path = os.path.join(
                        map_dir, '%s_%s.nii' % (map_type, contrast_id))
                    print("\t\tWriting %s ..." % map_path)

                    nibabel.save(out_map, map_path)

                    # collect zmaps for contrasts we're interested in
                    if map_type == 'z':
                        z_maps[contrast_id] = map_path

                    if map_type == 'effects':
                        effects_maps[contrast_id] = map_path

            return effects_maps, z_maps, mask_path, map_dirs
        except:
            return None

    # compute native-space maps and mask
    stuff = mem.cache(tortoise)(
        subject_data.func, subject_data.anat)
    if stuff is None:
        return None
    effects_maps, z_maps, mask_path, map_dirs = stuff

    # remove repeated contrasts
    contrasts = dict((cid, cval[0]) for cid, cval in contrasts.items())
    import json
    json.dump(dict((k, list(v)) for k, v in contrasts.items()),
              open(os.path.join(subject_data.tmp_output_dir,
                                "contrasts.json"), "w"))
    subject_data.contrasts = contrasts

    if normalize:
        assert hasattr(subject_data, "parameter_file")

        subject_data.native_effects_maps = effects_maps
        subject_data.native_z_maps = z_maps
        subject_data.native_mask_path = mask_path

        # warp effects maps and mask from native to standard space (MNI)
        apply_to_files = [
            v for _, v in subject_data.native_effects_maps.items()
            ] + [subject_data.native_mask_path]
        tmp = nipype_mem.cache(spm.Normalize)(
            parameter_file=getattr(subject_data, "parameter_file"),
            apply_to_files=apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=func_write_voxel_sizes,
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write',
            ignore_exception=False,
            ).outputs.normalized_files

        subject_data.mask = hard_link(tmp[-1], subject_data.output_dir)
        subject_data.effects_maps = dict(zip(effects_maps.keys(), hard_link(
                    tmp[:-1], map_dirs["effects"])))

        # warp anat image
        subject_data.anat = hard_link(nipype_mem.cache(spm.Normalize)(
                parameter_file=getattr(subject_data, "parameter_file"),
                apply_to_files=subject_data.anat,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=anat_write_voxel_sizes,
                write_wrap=[0, 0, 0],
                write_interp=1,
                jobtype='write',
                ignore_exception=False,
                ).outputs.normalized_files, subject_data.anat_output_dir)
    else:
        subject_data.mask = mask_path
        subject_data.effects_maps = effects_maps
        subject_data.z_maps = z_maps

    return subject_data

if __name__ == '__main__':
    ###########################################################################
    # CONFIGURATION
    protocols = [
        'WM',
        'MOTOR',
        'LANGUAGE',
        'EMOTION',
        'GAMBLING',
        'RELATIONAL',
        'SOCIAL'
        ]
    slicer = 'ortho'  # slicer of activation maps QA
    cut_coords = None
    threshold = 3.
    cluster_th = 15  # minimum number of voxels in reported clusters

    ####################################
    # read input configuration
    conf_file = os.path.join(os.path.dirname(sys.argv[0]), "HCP.ini")

    for protocol in protocols:
        subjects, preproc_params = _generate_preproc_pipeline(
            conf_file, protocol=protocol)

        fwhm = preproc_params.get("fwhm")
        task_output_dir = os.path.join(os.path.dirname(subjects[0].output_dir))
        kwargs = {"regress_motion": True,
                  "slicer": slicer,
                  "threshold": threshold,
                  "cluster_th": cluster_th,
                  "protocol": protocol,
                  "dc": not preproc_params.get(
                "disable_distortion_correction", False),
                  "realign": preproc_params["realign"],
                  "coregister": preproc_params["coregister"],
                  "segment": preproc_params["segment"],
                  "normalize": preproc_params["normalize"],
                  'func_write_voxel_sizes': preproc_params[
                'func_write_voxel_sizes'],
                  'anat_write_voxel_sizes': preproc_params[
                'anat_write_voxel_sizes'],
                  "fwhm": fwhm
                  }
        n_jobs = int(os.environ.get('N_JOBS', 1))
        if n_jobs > 1:
            subjects = Parallel(
                n_jobs=n_jobs, verbose=100)(delayed(
                    run_suject_level1_glm)(
                        subject_data,
                        **kwargs) for subject_data in subjects)
        else:
            subjects = [run_suject_level1_glm(subject_data,
                                              **kwargs)
                        for subject_data in subjects]
        subjects = [subject for subject in subjects if subject]

        # level 2
        stats_start_time = pretty_time()
        mask_images = [subject_data.mask for subject_data in subjects]
        group_mask = nibabel.Nifti1Image(
            intersect_masks(mask_images).astype(np.int8),
            nibabel.load(mask_images[0]).get_affine())
        nibabel.save(group_mask, os.path.join(
                task_output_dir, "mask.nii.gz"))

        print("... done.\r\n")
        print("Group GLM")
