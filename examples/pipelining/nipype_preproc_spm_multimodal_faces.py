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
from nilearn.glm.first_level.design_matrix import (make_first_level_design_matrix,
                                                         check_design_matrix)
from nilearn.glm.first_level import FirstLevelModel
import pandas as pd

# pypreprocess imports
from pypreprocess.datasets import fetch_spm_multimodal_fmri
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
hfcut = 1. / 128

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

    _duration = 0.6
    duration = _duration * np.ones(len(conditions))

    # build design matrix
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    paradigm = pd.DataFrame({'trial_type': conditions, 'duration': duration, 'onset': onsets})
    design_matrix = make_first_level_design_matrix(frametimes, paradigm,
                                       hrf_model=hrf_model,
                                       drift_model=drift_model,
                                       high_pass=hfcut)
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
    design_matrices=design_matrices)

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
    z_map = fmri_glm.compute_contrast(
        [contrast_val] * 2, output_type='z_score')

    z_maps[contrast_id] = z_map
