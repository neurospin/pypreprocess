import os
import time

import numpy as np
import pylab as pl
import nibabel
import scipy.io

from nipy.modalities.fmri.experimental_paradigm import (
                BlockParadigm, EventRelatedParadigm)
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel

from ..io_utils import load_4D_img, load_vol
from ..reporting.glm_reporter import generate_subject_stats_report
from ..reporting.base_reporter import ProgressReport


def execute_spm_auditory_glm(data, reg_motion=False):
    reg_motion = reg_motion and 'realignment_parameters' in data

    tr = 7.
    n_scans = 96
    _duration = 6
    epoch_duration = _duration * tr
    conditions = ['rest', 'active'] * 8
    duration = epoch_duration * np.ones(len(conditions))
    onset = np.linspace(0, (len(conditions) - 1) * epoch_duration,
                        len(conditions))
    paradigm = BlockParadigm(con_id=conditions, onset=onset, duration=duration)
    hfcut = 2 * 2 * epoch_duration

    # construct design matrix
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    drift_model = 'Cosine'
    hrf_model = 'Canonical With Derivative'

    add_reg_names = None
    add_regs = None
    if reg_motion:
        add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        add_regs = data['realignment_parameters'][0]
        if isinstance(add_regs, basestring):
            add_regs = np.loadtxt(add_regs)

    design_matrix = make_dmtx(frametimes,
                              paradigm, hrf_model=hrf_model,
                              drift_model=drift_model, hfcut=hfcut,
                              add_reg_names=add_reg_names,
                              add_regs=add_regs)

    # plot and save design matrix
    ax = design_matrix.show()
    ax.set_position([.05, .25, .9, .65])
    ax.set_title('Design matrix')
    dmat_outfile = os.path.join(data['output_dir'],
                                'design_matrix.png')
    pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)
    pl.close()

    # specify contrasts
    contrasts = {}
    n_columns = len(design_matrix.names)
    for i in range(paradigm.n_conditions):
        contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

    # more interesting contrasts"""
    contrasts['active-rest'] = contrasts['active'] - contrasts['rest']

    # fit GLM
    print('\r\nFitting a GLM (this takes time)...')
    fmri_glm = FMRILinearModel(load_4D_img(data['func'][0]),
                               design_matrix.matrix,
                               mask='compute')

    fmri_glm.fit(do_scaling=True, model='ar1')

    # save computed mask
    mask_path = os.path.join(data['output_dir'], "mask.nii.gz")
    print "Saving mask image %s..." % mask_path
    nibabel.save(fmri_glm.mask, mask_path)

    # compute bg unto which activation will be projected
    anat_img = load_vol(data['anat'])

    anat = anat_img.get_data()

    if anat.ndim == 4:
        anat = anat[..., 0]

    anat_affine = anat_img.get_affine()

    print "Computing contrasts..."
    z_maps = {}
    for contrast_id, contrast_val in contrasts.items():
        print "\tcontrast id: %s" % contrast_id
        z_map, t_map, eff_map, var_map = fmri_glm.contrast(
            contrasts[contrast_id],
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True,
            )

        # store stat maps to disk
        for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                                  [z_map, t_map, eff_map, var_map]):
            map_dir = os.path.join(
                data['output_dir'], '%s_maps' % dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if contrast_id == 'active-rest' and dtype == "z":
                z_maps[contrast_id] = map_path

            print "\t\t%s map: %s" % (dtype, map_path)

        print

    # do stats report
    stats_report_filename = os.path.join(data['reports_output_dir'],
                                         "report_stats.html")
    contrasts = dict((contrast_id, contrasts[contrast_id])
                     for contrast_id in z_maps.keys())
    generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        z_maps,
        fmri_glm.mask,
        design_matrices=[design_matrix],
        subject_id=data['subject_id'],
        anat=anat,
        anat_affine=anat_affine,
        cluster_th=50,  # we're only interested in this 'large' clusters

        # additional ``kwargs`` for more informative report
        paradigm=paradigm.__dict__,
        TR=tr,
        n_scans=n_scans,
        hfcut=hfcut,
        frametimes=frametimes,
        drift_model=drift_model,
        hrf_model=hrf_model,
        )

    ProgressReport().finish_dir(data['output_dir'])

    print "\r\nStatistic report written to %s\r\n" % stats_report_filename


def execute_spm_multimodal_fmri_glm(data, reg_motion=False):
    reg_motion = reg_motion and 'realignment_parameters' in data

    # experimental paradigm meta-params
    stats_start_time = time.ctime()
    tr = 2.
    drift_model = 'Cosine'
    hrf_model = 'Canonical With Derivative'
    hfcut = 128.

    # make design matrices
    design_matrices = []
    for x in range(2):
        n_scans = data['func'][x].shape[-1]

        timing = scipy.io.loadmat(data['trials_ses%i' % (x + 1)],
                                  squeeze_me=True, struct_as_record=False)

        faces_onsets = timing['onsets'][0].ravel()
        scrambled_onsets = timing['onsets'][1].ravel()
        onsets = np.hstack((faces_onsets, scrambled_onsets))
        onsets *= tr  # because onsets were reporting in 'scans' units
        conditions = ['faces'] * len(faces_onsets) + ['scrambled'] * len(
            scrambled_onsets)
        paradigm = EventRelatedParadigm(conditions, onsets)
        frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)

        add_reg_names = None
        add_regs = None
        if reg_motion:
            add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
            add_regs = np.loadtxt(data['realignment_parameters'][x])
            if isinstance(add_regs):
                add_regs = np.loadtxt(add_regs)
        design_matrix = make_dmtx(
            frametimes,
            paradigm, hrf_model=hrf_model,
            drift_model=drift_model, hfcut=hfcut,
            add_reg_names=add_reg_names,
            add_regs=add_regs
            )

        design_matrices.append(design_matrix)

    # specify contrasts
    contrasts = {}
    n_columns = len(design_matrix.names)
    for i in range(paradigm.n_conditions):
        contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

    # more interesting contrasts
    contrasts['faces-scrambled'] = contrasts['faces'] - contrasts['scrambled']
    contrasts['scrambled-faces'] = contrasts['scrambled'] - contrasts['faces']
    contrasts['effects_of_interest'] = contrasts[
        'faces'] + contrasts['scrambled']

    # we've thesame contrasts over sessions, so let's replicate
    contrasts = dict((contrast_id, [contrast_val] * 2)
                     for contrast_id, contrast_val in contrasts.items())

    # fit GLM
    print('\r\nFitting a GLM (this takes time)...')
    fmri_glm = FMRILinearModel([load_4D_img(sess_func)
                                for sess_func in data['func']],
                               [dmat.matrix for dmat in design_matrices],
                               mask='compute')
    fmri_glm.fit(do_scaling=True, model='ar1')

    # save computed mask
    mask_path = os.path.join(data['output_dir'], "mask.nii.gz")
    print "Saving mask image %s" % mask_path
    nibabel.save(fmri_glm.mask, mask_path)

    # compute bg unto which activation will be projected
    anat_img = load_vol(data['anat'])

    anat = anat_img.get_data()

    if anat.ndim == 4:
        anat = anat[..., 0]

    anat_affine = anat_img.get_affine()

    print "Computing contrasts .."
    z_maps = {}
    for contrast_id, contrast_val in contrasts.items():
        print "\tcontrast id: %s" % contrast_id
        z_map, t_map, eff_map, var_map = fmri_glm.contrast(
            contrast_val,
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True,
            )

        # store stat maps to disk
        for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                                  [z_map, t_map, eff_map, var_map]):
            map_dir = os.path.join(
                data['output_dir'], '%s_maps' % dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if dtype == 'z':
                z_maps[contrast_id] = map_path

            print "\t\t%s map: %s" % (dtype, map_path)

    # do stats report
    data['stats_report_filename'] = os.path.join(data['reports_output_dir'],
                                                 "report_stats.html")
    contrasts = dict((contrast_id, contrasts[contrast_id])
                     for contrast_id in z_maps.keys())
    generate_subject_stats_report(
        data['stats_report_filename'],
        contrasts,
        z_maps,
        fmri_glm.mask,
        anat=anat,
        anat_affine=anat_affine,
        design_matrices=design_matrices,
        subject_id=data['subject_id'],
        cluster_th=15,  # we're only interested in this 'large' clusters
        start_time=stats_start_time,

        # additional ``kwargs`` for more informative report
        paradigm=paradigm.__dict__,
        TR=tr,
        n_scans=n_scans,
        hfcut=hfcut,
        frametimes=frametimes,
        drift_model=drift_model,
        hrf_model=hrf_model,
        )

    ProgressReport().finish_dir(data['reports_output_dir'])

    print "\r\nStatistic report written to %s\r\n" % data[
        'stats_report_filename']

    return data

