"""
Author: DOHMATOB Elvis Dopgima elvis[dot]dohmatob[at]inria[dot]fr
Synopsis: Demo script for nipy's GLM and inference + reporting
on FSL's FEEDS fMRI single-subject example data
"""

import os
import numpy as np
import pandas as pd
from nilearn.glm.first_level.design_matrix import (make_first_level_design_matrix,
                                                         check_design_matrix)
from nilearn.glm.first_level import FirstLevelModel
import nibabel
import time
from pypreprocess.nipype_preproc_spm_utils import (SubjectData,
                                                   do_subjects_preproc)
from pypreprocess.datasets import fetch_fsl_feeds
from pypreprocess.io_utils import compute_mean_3D_image

"""MISC"""
DATASET_DESCRIPTION = "FSL FEEDS example data (single-subject)"

"""experimental setup"""
stats_start_time = time.ctime()
n_scans = 180
TR = 3.
EV1_epoch_duration = 2 * 30
EV2_epoch_duration = 2 * 45
TA = TR * n_scans
EV1_epochs = TA / EV1_epoch_duration
EV1_epochs = int(TA / EV1_epoch_duration)
EV2_epochs = int(TA / EV2_epoch_duration)
EV1_onset = np.linspace(0, EV1_epoch_duration * (EV1_epochs - 1), EV1_epochs)
EV2_onset = np.linspace(0, EV2_epoch_duration * (EV2_epochs - 1), EV2_epochs)
EV1_on = 30
EV2_on = 45
conditions = ['EV1'] * EV1_epochs + ['EV2'] * EV2_epochs
onset = list(EV1_onset) + list(EV2_onset)
duration = [EV1_on] * EV1_epochs + [EV2_on] * EV2_epochs
paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onset,
                         'duration': duration})
frametimes = np.linspace(0, (n_scans - 1) * TR, n_scans)
maximum_epoch_duration = max(EV1_epoch_duration, EV2_epoch_duration)
hfcut = 1.5 * maximum_epoch_duration  # why ?
hfcut = 1./hfcut

"""construct design matrix"""
drift_model = 'Cosine'
hrf_model = 'spm + derivative'
design_matrix = make_first_level_design_matrix(frame_times=frametimes,
                                   events=paradigm,
                                   hrf_model=hrf_model,
                                   drift_model=drift_model,
                                   high_pass=hfcut)

"""fetch input data"""
_subject_data = fetch_fsl_feeds()
subject_data = SubjectData()
subject_data.subject_id = "sub001"
subject_data.func = _subject_data.func
subject_data.anat = _subject_data.anat

output_dir = os.path.join(_subject_data.data_dir, "pypreprocess_output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
subject_data.output_dir = os.path.join(
    output_dir, subject_data.subject_id)



"""preprocess the data"""
results = do_subjects_preproc(
    [subject_data],
    output_dir=output_dir,
    dataset_id="FSL FEEDS single-subject",
    dataset_description=DATASET_DESCRIPTION,
    do_shutdown_reloaders=False,
    )

"""collect preprocessed data"""
fmri_files = results[0]['func']
anat_file = results[0]['anat']

"""specify contrasts"""
_, matrix, names = check_design_matrix(design_matrix)
contrasts = {}
n_columns = len(names)
I = np.eye(len(names))
for i in range(2):
    contrasts['%s' % names[2 * i]] = I[2 * i]

"""more interesting contrasts"""
contrasts['EV1>EV2'] = contrasts['EV1'] - contrasts['EV2']
contrasts['EV2>EV1'] = contrasts['EV2'] - contrasts['EV1']
contrasts['effects_of_interest'] = contrasts['EV1'] + contrasts['EV2']

"""fit GLM"""
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FirstLevelModel()
fmri_glm.fit(fmri_files, design_matrices=design_matrix)

"""save computed mask"""
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print("Saving mask image %s" % mask_path)
nibabel.save(fmri_glm.masker_.mask_img_, mask_path)

# compute bg unto which activation will be projected
mean_fmri_files = compute_mean_3D_image(fmri_files)
print("Computing contrasts ..")
z_maps = {}
for contrast_id, contrast_val in contrasts.items():
    print("\tcontrast id: %s" % contrast_id)
    z_map = fmri_glm.compute_contrast(
        contrasts[contrast_id], output_type='z_score')

    z_maps[contrast_id] = z_map
