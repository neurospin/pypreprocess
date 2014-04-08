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
from nilearn.image import resample_img, smooth_img
from nilearn.mass_univariate import permuted_ols

BET = True
FWHM = 0
DOWNSAMPLE_FACTOR = 1

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
nwbv = ext_vars['nwbv'].astype(float).reshape((-1, 1))
etiv = ext_vars['etiv'].astype(float).reshape((-1, 1))

# filter elderly subjects (= subjects with available Alzheimer info)
elderly_subjects_ids = np.where(educ != -1)[0]
images = [x for (i, x) in enumerate(images) if i in elderly_subjects_ids]
age = age[elderly_subjects_ids]
gender = gender[elderly_subjects_ids]
educ = educ[elderly_subjects_ids]
ses = ses[elderly_subjects_ids]
mmse = mmse[elderly_subjects_ids]
nwbv = nwbv[elderly_subjects_ids]
etiv = etiv[elderly_subjects_ids]
cdr = LabelBinarizer().fit_transform(
    LabelEncoder().fit_transform(ext_vars['cdr'][elderly_subjects_ids]))
#cdr[:, 0] += cdr[:, 1]
#cdr = cdr[:, [0, 2]]
#cdr = cdr[:, 2].reshape((-1, 1))
covars = np.hstack((age, gender))
tested_vars = np.hstack((age, gender, ses, mmse, cdr, etiv, nwbv, educ))

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
# resample images
new_affine = np.asarray(nibabel.load(images[0]).get_affine())
new_affine[:, :-1] *= DOWNSAMPLE_FACTOR
images_resampled = [resample_img(
        img,
        target_shape=(np.asarray(nibabel.load(images[0]).shape)
                      / DOWNSAMPLE_FACTOR).astype(int),
        target_affine=new_affine)
                    for img in nonnan_images]
images_resampled[0].to_filename("resampled_img.nii.gz")
print "Nifti masker"
# remove features with zero between-subject variance
images_masked = nifti_masker.fit_transform(images_resampled)
images_masked[:, images_masked.var(0) < 0.01] = 0.
# final masking
new_images = nifti_masker.inverse_transform(images_masked)
images_masked = nifti_masker.fit_transform(new_images)
n_samples, n_features = images_masked.shape
print n_samples, "subjects, ", n_features, "features"

"""
### Ward clustering on data
print "Ward clustering"
mask_shape = nifti_masker.mask_img_.shape
connectivity = image.grid_to_graph(
    n_x=mask_shape[0], n_y=mask_shape[1], n_z=mask_shape[2],
    mask=np.asarray(nifti_masker.mask_img_.get_data()).astype(bool))
ward = WardAgglomeration(n_clusters=1000, connectivity=connectivity)
images_masked = ward.fit(images_masked[:5]).transform(images_masked)
"""

### Perform massively univariate analysis with permuted OLS ###################
print "Massively univariate model"
neg_log_pvals, all_scores, _ = permuted_ols(
    tested_vars, images_masked,  # + intercept as a covariate by default
    n_perm=1000,
    n_jobs=-1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals)

### Show results
print "Plotting results"
mean_anat = smooth_img(images[0], FWHM).get_data()
for img in images[1:]:
    mean_anat += smooth_img(img, FWHM).get_data()
mean_anat /= float(len(images))
ref_img = nibabel.load(images[0])
neg_log_pvals_unmasked_orig_sampling = resample_img(
    neg_log_pvals_unmasked, ref_img.get_affine(), ref_img.shape,
    interpolation='nearest')
picked_slice = 50
vmin = -np.log10(0.1)  # 10% corrected
plt.figure()
p_ma = np.ma.masked_less(neg_log_pvals_unmasked_orig_sampling.get_data(), vmin)
plt.imshow(np.rot90(mean_anat[..., picked_slice]),
           interpolation='nearest', cmap=plt.cm.gray, vmin=0., vmax=1.)
im = plt.imshow(np.rot90(p_ma[..., picked_slice, 1]),
                interpolation='nearest', cmap=plt.cm.autumn,
                vmin=vmin, vmax=max(np.amax(neg_log_pvals), 1.00001))
plt.axis('off')
plt.colorbar(im)
plt.subplots_adjust(0., 0.03, 1., 0.83)

plt.show()
