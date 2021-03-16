"""
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr
Synopsis: Minimal script for preprocessing single-subject data
"""

import os
import time
import numpy as np
import nibabel
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc
from pypreprocess.datasets import fetch_spm_auditory
import pandas as pd
from nilearn.glm.first_level.design_matrix import (make_first_level_design_matrix,
                                                         check_design_matrix)

from nilearn.plotting.matrix_plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
import matplotlib.pyplot as plt

# file containing configuration for preprocessing the data
this_dir = os.path.dirname(os.path.abspath(__file__))
jobfile = os.path.join(this_dir, "spm_auditory_preproc.ini")

# fetch spm auditory data
sd = fetch_spm_auditory()
dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(sd.anat)))

# construct experimental paradigm
stats_start_time = time.ctime()
tr = 7.
n_scans = 96
_duration = 6
n_conditions = 2
epoch_duration = _duration * tr
conditions = ['rest', 'active'] * 8
duration = epoch_duration * np.ones(len(conditions))
onset = np.linspace(0, (len(conditions) - 1) * epoch_duration,
                    len(conditions))
paradigm = pd.DataFrame(
    {'onset': onset, 'duration': duration, 'trial_type': conditions})

hfcut = 2 * 2 * epoch_duration
hfcut = 1./hfcut

fd = open(sd.func[0].split(".")[0] + "_onset.txt", "w")
for c, o, d in zip(conditions, onset, duration):
    fd.write("%s %s %s\r\n" % (c, o, d))
fd.close()

# preprocess the data
subject_data = do_subjects_preproc(jobfile, dataset_dir=dataset_dir)[0]

# construct design matrix
nscans = len(subject_data.func[0])
frametimes = np.linspace(0, (nscans - 1) * tr, nscans)
drift_model = 'Cosine'
hrf_model = 'spm + derivative'
design_matrix = make_first_level_design_matrix(
    frametimes, paradigm, hrf_model=hrf_model, drift_model=drift_model,
    high_pass=hfcut)

# plot and save design matrix
ax = plot_design_matrix(design_matrix)
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')
dmat_outfile = os.path.join(subject_data.output_dir, 'design_matrix.png')
plt.savefig(dmat_outfile, bbox_inches="tight", dpi=200)

# specify contrasts
contrasts = {}
_, matrix, names = check_design_matrix(design_matrix)
contrast_matrix = np.eye(len(names))
for i in range(len(names)):
    contrasts[names[i]] = contrast_matrix[i]

# more interesting contrasts"""
contrasts = {'active-rest': contrasts['active'] - contrasts['rest']}

# fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FirstLevelModel(noise_model='ar1', standardize=False, t_r=tr).fit(
    [nibabel.concat_images(subject_data.func[0])], design_matrices=design_matrix)


# save computed mask
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print("Saving mask image %s" % mask_path)
nibabel.save(fmri_glm.masker_.mask_img_, mask_path)

# compute bg unto which activation will be projected
anat_img = nibabel.load(subject_data.anat)

print("Computing contrasts ..")
z_maps = {}
effects_maps = {}
for contrast_id, contrast_val in contrasts.items():
    print("\tcontrast id: %s" % contrast_id)
    z_map = fmri_glm.compute_contrast(
        contrasts[contrast_id], output_type='z_score')

    z_maps[contrast_id] = z_map
