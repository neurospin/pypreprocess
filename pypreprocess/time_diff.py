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
import numpy as np

import nibabel as nib

from nilearn.plotting import plot_stat_map
from nilearn._utils import check_niimgs
from nilearn.image import mean_img


def multi_session_time_slice_diffs(img_list):
    """ time slice difference on several 4D images

    Parameters
    ----------
    img_list: list of 4D Niimg-like
        Input multi-session images

    returns
    -------
    results : dict
        see time_slice_diffs docstring for details.

    note
    ----
    The results are accumulated across sessions
    """
    results = {}
    for i, img in enumerate(img_list):
        results_ = time_slice_diffs(img)
        if i == 0:
            for key, val in results_.items():
                results[key] = val
        else:
            results['volume_mean_diff2'] = np.hstack((
                    results['volume_mean_diff2'],
                    results_['volume_mean_diff2']))
            results['slice_mean_diff2'] = np.vstack((
                    results['slice_mean_diff2'],
                    results_['slice_mean_diff2']))
            results['volume_means'] = np.hstack((
                    results['volume_means'],
                    results_['volume_means']))
            results['diff2_mean_vol'] = mean_img(
                [results['diff2_mean_vol'], results_['diff2_mean_vol']])
            results['slice_diff2_max_vol'] = nib.Nifti1Image(
                np.maximum(results_['slice_diff2_max_vol'].get_data(),
                           results['slice_diff2_max_vol'].get_data()),
                results['slice_diff2_max_vol'].get_affine()
                )
    return results


def time_slice_diffs(img):
    ''' Time-point to time-point differences over volumes and slices

    We think of the passed array as an image.
    The last dimension is assumed to be time.

    Parameters
    ----------
    img: 4D Niimg-like
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


def plot_tsdiffs(results, use_same_figure=True):
    ''' Plotting routine for time series difference metrics

    Requires matplotlib

    Parameters
    ----------
    results : dict
        Results of format returned from
        :func:`pypreprocess.time_diff.time_slice_diff`

    use_same_figure : bool
        Whether to put all the plots on the same figure. If False, one
        figure will be created for each plot.

    '''
    import matplotlib.pyplot as plt

    T = len(results['volume_means'])
    S = results['slice_mean_diff2'].shape[1]
    mean_means = np.mean(results['volume_means'])
    scaled_slice_diff = results['slice_mean_diff2'] / mean_means

    n_plots = 6

    if use_same_figure:
        fig, axes = plt.subplots((n_plots + 1) // 2, 2)
        # Slightly easier to flatten axes to treat the
        # use_same_figure=False case in a similar fashion
        axes = axes.T.reshape(-1)
        fig.set_size_inches(12, 6, forward=True)
        fig.subplots_adjust(top=0.97, bottom=0.08, left=0.1, right=0.98, hspace=0.3, wspace=0.18)
    else:
        axes = [plt.figure(figsize=(6, 2)).add_subplot(111) for __ in range(n_plots)]

    def xmax_labels(ax, val, xlabel, ylabel):
        xlims = ax.axis()
        ax.axis((0, val) + xlims[2:])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    iter_axes = iter(axes)

    # plot of mean volume variance
    ax = next(iter_axes)
    ax.plot(results['volume_mean_diff2'] / mean_means)
    xmax_labels(ax, T - 1, 'Difference image number', 'Scaled variance')

    # plot of diff by slice
    ax = next(iter_axes)
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
    ax = next(iter_axes)
    ax.plot(results['volume_means'] / mean_means)
    xmax_labels(ax, T,
                'Image number',
                'Scaled mean \n voxel intensity')

    # slice plots min max mean
    ax = next(iter_axes)
    ax.hold(True)
    ax.plot(np.mean(scaled_slice_diff, 0), 'k')
    ax.plot(np.min(scaled_slice_diff, 0), 'b')
    ax.plot(np.max(scaled_slice_diff, 0), 'r')
    ax.hold(False)
    xmax_labels(ax, S+1,
                'Slice number',
                'Max/mean/min \n slice variation')

    ax = next(iter_axes)
    plot_stat_map(results['diff2_mean_vol'], bg_img=None,
                  display_mode='z', cut_coords=5,
                  axes=ax, black_bg=True, title='diff2_mean_vol')

    ax = next(iter_axes)
    plot_stat_map(results['slice_diff2_max_vol'], bg_img=None,
                  display_mode='z', cut_coords=5,
                  axes=ax, black_bg=True, title='slice_diff2_max_vol')

    return axes


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from nilearn import datasets

    nyu_rest_dataset = datasets.fetch_nyu_rest(n_subjects=2)
    filenames = nyu_rest_dataset.func
    results = multi_session_time_slice_diffs(filenames)
    #results = time_slice_diffs(filename)
    plot_tsdiffs(results)
    plot_tsdiffs(results, use_same_figure=False)
    plt.show()
