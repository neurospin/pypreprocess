"""
VBM on Oasis data.

"""
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

BET = True

### Gather data
# images
path_to_images = "/home/virgile/wip/retreat/pypreprocess_output"
images = glob.glob(
    os.path.join(path_to_images,
                 "OAS1_*_MR1/mwc1OAS1_*dim%s.nii" % ("bet" if BET else "")))

n_samples = len(images)
# explanatory variates
path_to_csv = "/home/virgile/wip/retreat/oasis/oasis_cross-sectional.csv"
ext_vars = np.recfromcsv(path_to_csv)[:n_samples]
age = ext_vars['age'].astype(float).reshape((-1, 1))
gender = LabelEncoder().fit_transform(
    ext_vars['mf']).astype(float).reshape((-1, 1))
educ = ext_vars['educ'].astype(float).reshape((-1, 1))
ses = ext_vars['ses'].astype(float).reshape((-1, 1))
mmse = ext_vars['mmse'].astype(float).reshape((-1, 1))

# filter elderly subjects (= subjects with available Alzheimer info)
elderly_subjects_ids = np.where(educ != -1)[0]
images = [x for (i, x) in enumerate(images) if i in elderly_subjects_ids]
age = age[elderly_subjects_ids]
gender = gender[elderly_subjects_ids]
educ = educ[elderly_subjects_ids]
ses = ses[elderly_subjects_ids]
mmse = mmse[elderly_subjects_ids]
cdr = LabelBinarizer().fit_transform(
    LabelEncoder().fit_transform(ext_vars['cdr'][elderly_subjects_ids]))
covars = np.hstack((age, gender))

### Mask data
nifti_masker = NiftiMasker(
    memory='nilearn_cache',
    memory_level=1)  # cache options
images_masked = nifti_masker.fit_transform(images)
images_masked[:, images_masked.var(0) == 0] = 0.
new_images = nifti_masker.inverse_transform(images_masked)
images_masked = nifti_masker.fit_transform(new_images)
n_samples, n_features = images_masked.shape

### Perform massively univariate analysis with permuted OLS ###################
neg_log_pvals, all_scores, _ = permuted_ols(
    mmse, images_masked, covars,  # + intercept as a covariate by default
    n_perm=1000,
    n_jobs=-1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals).get_data()

### Show results
picked_slice = 38
vmin = -np.log10(0.1)  # 10% corrected
plt.figure()
p_ma = np.ma.masked_less(neg_log_pvals_unmasked, vmin)
plt.imshow(np.rot90(p_ma[..., picked_slice, 0]),
           interpolation='nearest', cmap=plt.cm.gray,
           vmin=vmin, vmax=max(np.amax(neg_log_pvals), 1.00001))

plt.show()
