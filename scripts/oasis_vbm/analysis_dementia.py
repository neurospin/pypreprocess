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
import nibabel
from nilearn.input_data import NiftiMasker
from nilearn.image import smooth_img
from nilearn.mass_univariate import permuted_ols

FWHM = 5

### Gather data
# images
path_to_images = "/home/virgile/wip/retreat/pypreprocess_output"
images = sorted(glob.glob(
        os.path.join(path_to_images, "OAS1_*_MR1/mwc2OAS1_*dimbet.nii")))
#images = images[:39]  # disc1 only

n_samples = len(images)
# explanatory variates
path_to_csv = "/home/virgile/wip/retreat/oasis/oasis_cross-sectional.csv"
ext_vars = np.recfromcsv(path_to_csv)[:n_samples]
age = ext_vars['age'].astype(float).reshape((-1, 1))

# filter elderly subjects (= subjects with available Alzheimer info)
elderly_subjects_ids = np.where(~np.isnan(ext_vars['cdr']))[0]
images = [x for (i, x) in enumerate(images) if i in elderly_subjects_ids]
age = age[elderly_subjects_ids]
cdr = LabelBinarizer().fit_transform(
    LabelEncoder().fit_transform(ext_vars['cdr'][elderly_subjects_ids]))
cdr = cdr[:, -1].reshape((-1, 1))  # build impairment variate

### Mask data
print "Resample images"
nifti_masker = NiftiMasker(
    smoothing_fwhm=FWHM,
    memory='nilearn_cache',
    memory_level=1)  # cache options
# remove NaNs from images
ref_affine = np.asarray(nibabel.load(images[0]).get_affine())
images_ = [np.asarray(nibabel.load(img).get_data()) for img in images]
nonnan_images = []
for img in images_:
    img[np.isnan(img)] = 0.
    nonnan_images.append(nibabel.Nifti1Image(img, ref_affine))
print "Nifti masker"
# remove features with zero between-subject variance
images_masked = nifti_masker.fit_transform(images)
images_masked[:, images_masked.var(0) < 0.01] = 0.
# final masking
new_images = nifti_masker.inverse_transform(images_masked)
images_masked = nifti_masker.fit_transform(new_images)
n_samples, n_features = images_masked.shape
print n_samples, "subjects, ", n_features, "features"

### Perform massively univariate analysis with permuted OLS ###################
print "Massively univariate model"
neg_log_pvals, all_scores, _ = permuted_ols(
    cdr, images_masked, age,  # + intercept as a covariate by default
    n_perm=1000,
    n_jobs=-1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals).get_data()[..., 0]

### Show results
print "Plotting results"
# background anat
mean_anat = smooth_img(images[0], FWHM).get_data()
for img in images[1:]:
    mean_anat += smooth_img(img, FWHM).get_data()
mean_anat /= float(len(images))
ref_img = nibabel.load(images[0])
picked_slice = 36
vmin = -np.log10(0.1)  # 10% corrected
plt.figure()
p_ma = np.ma.masked_less(neg_log_pvals_unmasked, vmin)
plt.imshow(np.rot90(mean_anat[..., picked_slice]),
           interpolation='nearest', cmap=plt.cm.gray, vmin=0., vmax=1.)
im = plt.imshow(np.rot90(p_ma[..., picked_slice]),
                interpolation='nearest', cmap=plt.cm.autumn,
                vmin=vmin, vmax=np.amax(neg_log_pvals_unmasked))
plt.axis('off')
plt.colorbar(im)
plt.subplots_adjust(0., 0.03, 1., 0.83)

plt.show()
