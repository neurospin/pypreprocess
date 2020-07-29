"""
:Author: yannick schwartz, dohmatob elvis dopgima
:Synopsis: Minimal script for preprocessing single-subject data
+ GLM with nistats
"""

# standard imports
import sys
import os
import time
import nibabel
import numpy as np
import scipy.io

# imports for GLM business
from nilearn.stats.first_level_model.design_matrix import (make_first_level_design_matrix,
                                                         check_design_matrix)
from nilearn.stats.first_level_model import FirstLevelModel
import pandas as pd

# pypreprocess imports
from pypreprocess.datasets import fetch_spm_multimodal_fmri
from pypreprocess.reporting.base_reporter import ProgressReport
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.nipype_preproc_spm_utils import do_subject_preproc
from pypreprocess.subject_data import SubjectData

# file containing configuration for preprocessing the data
this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
jobfile = os.path.join(this_dir, "multimodal_faces_preproc.ini")

# set dataset dir
if len(sys.argv) > 1:
    dataset_dir = sys.argv[1]
else:
    dataset_dir = os.path.join(this_dir, "spm_multimodal_faces")

# fetch spm multimodal_faces data
subject_data = fetch_spm_multimodal_fmri()
dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(
    subject_data.anat)))

# preprocess the data
subject_id = "sub001"
subject_data = SubjectData(
    output_dir=os.path.join(dataset_dir, "pypreprocess_output", subject_id),
    subject_id=subject_id, func=[subject_data.func1, subject_data.func2],
    anat=subject_data.anat, trials_ses1=subject_data.trials_ses1,
    trials_ses2=subject_data.trials_ses2, session_ids=["Session1", "Session2"])
subject_data = do_subject_preproc(subject_data, realign=True, coregister=True,
                                  segment=True, normalize=True)

# experimental paradigm meta-params
stats_start_time = time.ctime()
tr = 2.
drift_model = 'Cosine'
hrf_model = 'spm + derivative'
hfcut = 128.

# make design matrices
first_level_effects_maps = []
mask_images = []
design_matrices = []
for x in range(2):
    if not os.path.exists(subject_data.output_dir):
        os.makedirs(subject_data.output_dir)

    # build paradigm
    n_scans = len(subject_data.func[x])
    timing = scipy.io.loadmat(getattr(subject_data, "trials_ses%i" % (x + 1)),
                              squeeze_me=True, struct_as_record=False)

    faces_onsets = timing['onsets'][0].ravel()
    scrambled_onsets = timing['onsets'][1].ravel()
    onsets = np.hstack((faces_onsets, scrambled_onsets))
    onsets *= tr  # because onsets were reporting in 'scans' units
    conditions = ['faces'] * len(faces_onsets) + ['scrambled'] * len(
        scrambled_onsets)

    # build design matrix
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    paradigm = pd.DataFrame({'name': conditions, 'onset': onsets})
    design_matrix = make_first_level_design_matrix(frametimes, paradigm,
                                       hrf_model=hrf_model,
                                       drift_model=drift_model,
                                       period_cut=hfcut)
    design_matrices.append(design_matrix)

# specify contrasts
_, matrix, names = check_design_matrix(design_matrix)
contrasts = {}
n_columns = len(names)
contrast_matrix = np.eye(n_columns)
for i in range(2):
    contrasts[names[2 * i]] = contrast_matrix[2 * i]

# more interesting contrasts
contrasts['faces-scrambled'] = contrasts['faces'] - contrasts['scrambled']
contrasts['scrambled-faces'] = -contrasts['faces-scrambled']
contrasts['effects_of_interest'] = contrasts['faces'] + contrasts['scrambled']

# fit GLM
print('Fitting a GLM (this takes time)...')
fmri_glm = FirstLevelModel().fit(
    [nibabel.concat_images(x) for x in subject_data.func],
    design_matrices)

# save computed mask
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print("Saving mask image %s" % mask_path)
nibabel.save(fmri_glm.masker_.mask_img_, mask_path)
mask_images.append(mask_path)

# compute contrast maps
z_maps = {}
effects_maps = {}
for contrast_id, contrast_val in contrasts.items():
    print("\tcontrast id: %s" % contrast_id)
    z_map, t_map, effects_map, var_map = fmri_glm.transform(
        [contrast_val] * 2, contrast_name=contrast_id, output_z=True,
        output_stat=True, output_effects=True, output_variance=True)
    for map_type, out_map in zip(['z', 't', 'effects', 'variance'],
                              [z_map, t_map, effects_map, var_map]):
        map_dir = os.path.join(
            subject_data.output_dir, '%s_maps' % map_type)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(
            map_dir, '%s.nii.gz' % contrast_id)
        print("\t\tWriting %s ..." % map_path)
        nibabel.save(out_map, map_path)
        if map_type == 'z':
            z_maps[contrast_id] = map_path
        if map_type == 'effects':
            effects_maps[contrast_id] = map_path

# generate stats report
anat_img = nibabel.load(subject_data.anat)
stats_report_filename = os.path.join(subject_data.output_dir, "reports",
                                     "report_stats.html")
generate_subject_stats_report(
    stats_report_filename, contrasts, z_maps, fmri_glm.masker_.mask_img_,
    anat=anat_img, threshold=2.3, cluster_th=15,
    design_matrices=design_matrices, TR=tr,
    subject_id="sub001", start_time=stats_start_time, n_scans=n_scans,
    title="GLM for subject %s" % subject_data.subject_id, hfcut=hfcut,
    paradigm=paradigm, frametimes=frametimes,
    drift_model=drift_model, hrf_model=hrf_model)
ProgressReport().finish_dir(subject_data.output_dir)
print("Statistic report written to %s\r\n" % stats_report_filename)
