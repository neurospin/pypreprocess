"""
Outlier detection and removal for OASIS preprocessed data.

"""
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr. 2014
import os
import glob
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import nibabel
from sklearn.metrics import euclidean_distances
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img

BET = True
FWHM = 0
DOWNSAMPLE_FACTOR = 1.

### Gather data
path_to_images = "/home/virgile/wip/retreat/pypreprocess_output"
images = glob.glob(
    os.path.join(path_to_images,
                 "OAS1_*_MR1/mwc1OAS1_*dim%s.nii" % ("bet" if BET else "")))

### Mask data
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
images = [resample_img(
        img,
        target_shape=(np.asarray(nibabel.load(images[0]).shape)
                      / DOWNSAMPLE_FACTOR).astype(int),
        target_affine=new_affine)
          for img in nonnan_images]
images[0].to_filename("resmpled_img.nii.gz")
print "Nifti masker"
# remove features with zero between-subject variance
images_masked = nifti_masker.fit_transform(images)
images_masked[:, images_masked.var(0) < 0.01] = 0.
# final masking
new_images = nifti_masker.inverse_transform(images_masked)
images_masked = nifti_masker.fit_transform(new_images)
n_samples, n_features = images_masked.shape
print n_samples, "subjects, ", n_features, "features"


### Euclidean distance between subjects
dist = euclidean_distances(images_masked)
mahalanobis_dist = np.mean(dist, 0) - np.median(dist)
threshold = stats.chi2(n_samples).isf(0.1 / float(n_samples))

# plot summary
plt.figure()
plt.scatter(np.arange(n_samples), mahalanobis_dist)
plt.hlines(threshold, 0, n_samples, color="black", linestyle='--')
plt.xlabel("Subject IDs")
plt.ylabel("Average euclidean distance to other subjects")
plt.xlim(0, n_samples)

# show outliers
outlier_ids = np.where(mahalanobis_dist > threshold)[0]
outliers_unmasked = nifti_masker.inverse_transform(images_masked[outlier_ids])
n_outliers = outliers_unmasked.shape[-1]
if n_outliers > 0:
    picked_slice = 30
    grid = ImageGrid(plt.figure(), 111, nrows_ncols=(1, n_outliers),
                     direction="row",
                     axes_pad=0.05, add_all=True, label_mode="1",
                     share_all=True, cbar_location="right", cbar_mode="single",
                     cbar_size="7%", cbar_pad="1%")

    for i in np.arange(n_outliers):
        ax = grid[i]
        im = ax.imshow(
            np.rot90(outliers_unmasked.get_data()[..., picked_slice, i]),
            interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
        ax.set_title(str.split(images[outlier_ids[i]], "mwc1")[1][:10])
        ax.axis('off')

    grid[0].cax.colorbar(im)
else:
    print "No outlier found"

plt.show()
