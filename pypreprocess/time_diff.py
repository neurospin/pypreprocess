# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Time series diagnostics

These started life as ``tsdiffana.m`` - see
http://imaging.mrc-cbu.cam.ac.uk/imaging/DataDiagnostics

Oliver Josephs (FIL) gave Matthew Brett the idea of time-point to time-point
subtraction as a diagnostic for motion and other sudden image changes.
This has been implemented in the Nipy package.

We give here a simpler implementation with modified dependences

'''
import os

import numpy as np

import nibabel as nib

from nilearn.plotting import plot_stat_map
from nilearn._utils import check_niimgs


def time_slice_diffs(img):
    ''' Time-point to time-point differences over volumes and slices

    We think of the passed array as an image.
    The last dimension is assumed to be time.

    Parameters
    ----------
    img: 4D Niimg-like,
         the input (4D) image

    Returns
    -------
    results : dict

        ``T`` is the number of time points (``arr.shape[time_axis]``)

        ``S`` is the number of slices (``arr.shape[slice_axis]``)

        ``v`` is the shape of a volume (``rollimg(arr, time_axis)[0].shape``)

        ``d2[t]`` is the volume of squared differences between voxels at
        time point ``t`` and time point ``t+1``

        `results` has keys:

        * 'volume_mean_diff2' : (T-1,) array
           array containing the mean (over voxels in volume) of the
           squared difference from one time point to the next
        * 'slice_mean_diff2' : (T-1, S) array
           giving the mean (over voxels in slice) of the difference from
           one time point to the next, one value per slice, per
           timepoint
        * 'volume_means' : (T,) array
           mean over voxels for each volume ``vol[t] for t in 0:T``
        * 'slice_diff2_max_vol' : v[:] array
           volume, of same shape as input time point volumes, where each slice
           is is the slice from ``d2[t]`` for t in 0:T-1, that has the largest
           variance across ``t``. Thus each slice in the volume may well result
           from a different difference time point.
        * 'diff2_mean_vol`` : v[:] array
           volume with the mean of ``d2[t]`` across t for t in 0:T-1.

    '''
    img = check_niimgs(img)
    shape = img.shape
    T = shape[-1]
    S = shape[-2]  # presumably the slice axis -- to be reconsidered ?

    # loop over time points to save memory
    # initialize the results
    slice_squared_differences = np.empty((T - 1, S))
    vol_mean = np.empty((T,))
    diff_mean_vol = np.zeros(shape[:3])
    slice_diff_max_vol = np.zeros(shape[:3])
    slice_diff_max = np.zeros(S)
    arr = img.get_data()  # inefficient ??
    last_vol = arr[..., 0]
    vol_mean[0] = last_vol.mean()

    # loop over scans: increment statistics
    for vol_index in range(0, T - 1):
        current_vol = arr[..., vol_index + 1]  # shape vol_shape
        vol_mean[vol_index + 1] = current_vol.mean()
        squared_diff = (current_vol - last_vol) ** 2
        diff_mean_vol += squared_diff
        slice_squared_differences[vol_index] = squared_diff.mean(0).mean(0)
        # check whether we have found a highest-diff slice
        larger_diff = slice_squared_differences[vol_index] > slice_diff_max
        if any(larger_diff):
            slice_diff_max[larger_diff] =\
                slice_squared_differences[vol_index][larger_diff]
            slice_diff_max_vol[..., larger_diff] =\
                squared_diff[..., larger_diff]
        last_vol = current_vol
    vol_squared_differences = slice_squared_differences.mean(1)
    diff_mean_vol /= (T - 1)

    # Return the outputs as images
    affine = img.get_affine()
    diff2_mean_vol = nib.Nifti1Image(diff_mean_vol, affine)
    slice_diff2_max_vol = nib.Nifti1Image(slice_diff_max_vol, affine)
    return {'volume_mean_diff2': vol_squared_differences,
            'slice_mean_diff2': slice_squared_differences,
            'volume_means': vol_mean,
            'diff2_mean_vol': diff2_mean_vol,
            'slice_diff2_max_vol': slice_diff2_max_vol}


def plot_tsdiffs(results, axes=None):
    ''' Plotting routine for time series difference metrics

    Requires matplotlib

    Parameters
    ----------
    results : dict
       Results of format returned from
       :func:`pypreprocess.time_diff.time_slice_diff`
    '''
    import matplotlib.pyplot as plt

    T = len(results['volume_means'])
    S = results['slice_mean_diff2'].shape[1]
    mean_means = np.mean(results['volume_means'])
    scaled_slice_diff = results['slice_mean_diff2'] / mean_means

    if axes is None:
        fig, axes = plt.subplots(3, 2)
        fig.set_size_inches(12, 6, forward=True)

    def xmax_labels(ax, val, xlabel, ylabel):
        xlims = ax.axis()
        ax.axis((0, val) + xlims[2:])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # plot of mean volume variance
    ax = axes[0][0]
    ax.plot(results['volume_mean_diff2'] / mean_means)
    xmax_labels(ax, T - 1, 'Difference image number', 'Scaled variance')

    # plot of diff by slice
    ax = axes[1][0]
    # Set up the color map for the different slices:
    X, Y = np.meshgrid(np.arange(scaled_slice_diff.shape[0]),
                       np.arange(scaled_slice_diff.shape[1]))

    # Use HSV in order to code the slices from bottom to top:
    ax.scatter(X.T.ravel(), scaled_slice_diff.ravel(),
               c=Y.T.ravel(), cmap=plt.cm.hsv,
               alpha=0.2)

    xmax_labels(ax, T - 1,
                'Difference image number',
                'Slice by slice variance')

    # mean intensity
    ax = axes[2][0]
    ax.plot(results['volume_means'] / mean_means)
    xmax_labels(ax, T,
                'Image number',
                'Scaled mean \n voxel intensity')

    # slice plots min max mean
    ax = axes[0][1]
    ax.hold(True)
    ax.plot(np.mean(scaled_slice_diff, 0), 'k')
    ax.plot(np.min(scaled_slice_diff, 0), 'b')
    ax.plot(np.max(scaled_slice_diff, 0), 'r')
    ax.hold(False)
    xmax_labels(ax, S+1,
                'Slice number',
                'Max/mean/min \n slice variation')

    ax = axes[1][1]
    plot_stat_map(results['diff2_mean_vol'], bg_img=None,
                  display_mode='z', cut_coords=5,
                  axes=ax, black_bg=True, title='diff2_mean_vol')

    ax = axes[2][1]
    plot_stat_map(results['slice_diff2_max_vol'], bg_img=None,
                  display_mode='z', cut_coords=5,
                  axes=ax, black_bg=True, title='slice_diff2_max_vol')

    return axes


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filename = '~/nilearn_data/nyu_rest/session1/sub22894/func/lfo.nii.gz'
    filename = os.path.expanduser(filename)
    results = time_slice_diffs(filename)
    plot_tsdiffs(results)
    plt.show()
