import os
import numpy as np
import pylab as pl
import time
import nibabel
import scipy.io
import joblib
from nipy.modalities.fmri.experimental_paradigm import (
    BlockParadigm,
    EventRelatedParadigm)
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel

from ..io_utils import (load_4D_img,
                        load_vol
                        )
from ..datasets import (fetch_spm_auditory_data,
                       fetch_spm_multimodal_fmri_data
                       )
from ..reporting.glm_reporter import generate_subject_stats_report
from ..reporting.base_reporter import ProgressReport
from ..purepython_preproc_utils import do_subject_preproc


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

    # specify contrasts
    contrasts = {}
    n_columns = len(design_matrix.names)
    for i in xrange(paradigm.n_conditions):
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
    for contrast_id, contrast_val in contrasts.iteritems():
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
    stats_report_filename = os.path.join(data['output_dir'],
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
    for x in xrange(2):
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
    for i in xrange(paradigm.n_conditions):
        contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

    # more interesting contrasts
    contrasts['faces-scrambled'] = contrasts['faces'] - contrasts['scrambled']
    contrasts['scrambled-faces'] = contrasts['scrambled'] - contrasts['faces']
    contrasts['effects_of_interest'] = contrasts[
        'faces'] + contrasts['scrambled']

    # we've thesame contrasts over sessions, so let's replicate
    contrasts = dict((contrast_id, [contrast_val] * 2)
                     for contrast_id, contrast_val in contrasts.iteritems())

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
    for contrast_id, contrast_val in contrasts.iteritems():
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
    data['stats_report_filename'] = os.path.join(data['output_dir'],
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

    ProgressReport().finish_dir(data['output_dir'])

    print "\r\nStatistic report written to %s\r\n" % data[
        'stats_report_filename']

    return data

# if __name__ == '__main__':
#     sd1 = fetch_spm_auditory_data(os.path.join(os.environ['HOME'],
#                                               "CODE/datasets/spm_auditory"))
#     spm_auditory_subject_data = {'subject_id': 'sub001', 'func': [sd1.func],
#                                  'anat': sd1.anat,
#                                  'n_sessions': 1}

#     sd2 = fetch_spm_multimodal_fmri_data(os.path.join(
#             os.environ['HOME'],
#             "CODE/datasets/spm_multimodal_fmri"))
#     spm_multimodal_fmri_data = sd2.dictcopy()
#     spm_multimodal_fmri_data.update({'subject_id': 'sub001',
#                                     'func': [sd2.func1, sd2.func2],
#                                     'anat': sd2.anat,
#                                      'n_sessions': 2})
#     del spm_multimodal_fmri_data['func1']
#     del spm_multimodal_fmri_data['func2']

#     # run pipeline
#     _output_dir = os.path.abspath("single_subject_pipeline_runs")

#     def pipeline_factory(subject_data, output_dir):
#         """
#         Generates different pipelines.

#         """

#         for do_stc in [True, False]:
#             for do_realign in [True, False]:
#                 for reg_motion in [False, True]:
#                     if reg_motion and not do_realign:
#                         continue
#                     for fwhm in [None, [5., 5., 5.]]:
#                         pipeline_remark = ""
#                         pipeline_remark = "_with_stc" if do_stc else \
#                             "_without_stc"
#                         pipeline_remark += (
#                             ("_with_mc" + ("_with_reg_motion" if reg_motion \
#                                                else "_without_reg_motion"))
#                             ) if do_realign else "_without_mc"
#                         pipeline_remark += "_without_smoothing" if fwhm is \
#                             None else "_with_smoothing"

#                         subject_data['output_dir'] = os.path.join(
#                             _output_dir,
#                             output_dir,
#                             subject_data['subject_id']
#                             )

#                         print "\t\t\tpipeline: %s (output_dir = %s)" % (
#                             pipeline_remark, subject_data['output_dir'])

#                         yield (subject_data, do_stc, do_realign,


#                                reg_motion, fwhm,
#                                pipeline_remark)

#     def _pipeline_runner(subject_data, output_dir, execute_glm):
#         """
#         Runs a pipeline.

#         """

#         for (subject_data, do_stc, do_realign, reg_motion, fwhm,
#              stats_output_dir_basename) in pipeline_factory(subject_data,
#                                                             output_dir):
#             preproc_output = do_subject_preproc(
#                 subject_data,
#                 do_stc=do_stc,
#                 do_realign=do_realign,
#                 fwhm=fwhm,
#                 )

#             ########
#             # GLM
#             ########
#             preproc_output['output_dir'] = os.path.join(
#                 preproc_output['output_dir'], stats_output_dir_basename)
#             if not os.path.exists(preproc_output['output_dir']):
#                 os.makedirs(preproc_output['output_dir'])
#             return execute_glm(preproc_output, reg_motion=reg_motion)

#     # run pipelines
#     n_jobs = int(os.environ['N_JOBS']) if 'N_JOBS' in os.environ else -1
#     joblib.Parallel(n_jobs=n_jobs, verbose=100)(joblib.delayed(
#             _pipeline_runner)(subject_data, output_dir, execute_glm) for (
#             subject_data, output_dir, execute_glm) in zip(
#             [spm_multimodal_fmri_data,
#              spm_auditory_subject_data
#              ],
#             ['spm_multimodal_fmri',
#              'spm_auditory'
#              ],
#             [execute_spm_multimodal_fmri_glm,
#              execute_spm_auditory_glm
#              ]))
