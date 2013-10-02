"""
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

import os
import glob
import numpy as np
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
import nibabel
import time
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.reporting.base_reporter import ProgressReport
from pypreprocess.io_utils import compute_mean_3D_image
from pypreprocess.nipype_preproc_spm_utils_bis import (_do_subject_smooth,
                                                       SubjectData
                                                       )

# load data
DATA_DIR = "/home/elvis/CODE/datasets/connectome-db"
DATA_DIR2 = DATA_DIR + "-preproc"
SUBJECT_ID = "100307"
fmri_files = os.path.join(DATA_DIR2, SUBJECT_ID,
                          "wrtfMRI_MOTOR_3T_100307_unwarped.nii")
anat_file = os.path.join(DATA_DIR2, SUBJECT_ID,
                         "wT1w_acpc_dc_restore_brain.nii")
subject_data = SubjectData(func=fmri_files, output_dir="/tmp")
subject_data.nipype_results = {}
fmri_files = _do_subject_smooth(subject_data, [6., 6., 6.],
                                ).func

# build experimental paradigm
TR = 720. / 1000  # or .5 instead ?
conditions = []
onsets = []
durations = []
amplitudes = []
n_scans = nibabel.load(fmri_files).shape[-1]
for ev_filename in glob.glob(
    os.path.join(DATA_DIR, SUBJECT_ID, (
            "tfMRI_MOTOR/unprocessed/3T/tfMRI_MOTOR_LR"
            "/LINKED_DATA/EPRIME/EVs/*.txt")
                 )
    ):
    try:
        timing = np.loadtxt(ev_filename)
        if timing.ndim > 1:
            condition_name = os.path.basename(ev_filename).lower(
                ).split('.')[0]
            conditions = conditions + [condition_name] * timing.shape[0]
            onsets = onsets + list(timing[..., 0])
            durations = durations + list(timing[..., 1])
            amplitudes = amplitudes + list(timing[..., 2])
    except (OSError, IOError, TypeError, ValueError):
        continue

paradigm = BlockParadigm(con_id=conditions, onset=onsets, duration=durations,
                         amplitude=amplitudes)

# build design matrix
frametimes = np.linspace(0, (n_scans - 1) * TR, n_scans)
hfcut = 128
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
design_matrix = make_dmtx(frametimes,
                          paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

# specify contrasts
contrasts = {}
n_columns = len(design_matrix.names)
I = np.eye(len(design_matrix.names))
for i in xrange(paradigm.n_conditions):
    contrasts[design_matrix.names[2 * i]] = I[2 * i]

# more interesting contrasts
contrasts["effects_of_interest"] = np.array(contrasts.values()).sum(axis=0)
contrasts["lh - rh"] = contrasts['lh'] - contrasts['rh']
contrasts["lh - lf"] = contrasts['lh'] - contrasts['lf']
contrasts["rh - rf"] = contrasts['rh'] - contrasts['rf']
contrasts["lf - rf"] = contrasts['lf'] - contrasts['rf']
contrasts["hands - feet"] = contrasts['lh'] + contrasts[
    'rh'] - contrasts['lf'] - contrasts['rf']
contrasts["left - right"] = contrasts['lf'] + contrasts[
    'lf'] - contrasts['rh'] - contrasts['rf']

# fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FMRILinearModel(fmri_files, design_matrix.matrix,
           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

"""save computed mask"""
output_dir = '/tmp/kirikou/'
mask_path = os.path.join(output_dir, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)

# compute bg unto which activation will be projected
mean_fmri_files = compute_mean_3D_image(fmri_files)

anat_img = nibabel.load(anat_file)
anat = anat_img.get_data()
anat_affine = anat_img.get_affine()

print "Computing contrasts .."
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
            output_dir, '%s_maps' % dtype)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(
            map_dir, '%s.nii.gz' % contrast_id)
        nibabel.save(out_map, map_path)

        if dtype == "z":
            z_maps[contrast_id] = map_path

        print "\t\t%s map: %s" % (dtype, map_path)

    print

# do stats report
stats_report_filename = os.path.join(output_dir,
                                     "report_stats.html")
contrasts = dict((contrast_id, contrasts[contrast_id])
                 for contrast_id in z_maps.keys())
generate_subject_stats_report(
    stats_report_filename,
    contrasts,
    z_maps,
    fmri_glm.mask,
    design_matrices=[design_matrix],
    subject_id=SUBJECT_ID,
    anat=anat,
    anat_affine=anat_affine,
    threshold=3.,
    cluster_th=50,  # we're only interested in this 'large' clusters
    start_time=time.time(),

    # additional ``kwargs`` for more informative report
    paradigm=paradigm.__dict__,
    TR=TR,
    n_scans=n_scans,
    hfcut=hfcut,
    frametimes=frametimes,
    drift_model=drift_model,
    hrf_model=hrf_model,
    slicer='x',
    )

# shutdown main report page
ProgressReport().finish_dir(output_dir)

print "\r\nStatistic report written to %s\r\n" % stats_report_filename
