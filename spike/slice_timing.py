"""
:Module: slice_timing
:Synopsis: Module for STC
:Author: elvis[dot]dohmatob[at]inria[dot]fr

"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy


def hanning_window(t, L):
    """Since sinc(t) decays to zero as t departs from zero, in the
    sinc interpolation sum for BOLD(t0) taking into account all acquisitions,
    terms corresponding to acquisition times beyond a radius L of t can be
    dropped (their contribution to the said sum will be small). One way to do
    this is to convolve the sum with a Hanning window of appropriate width
    about t0. However, mind you the way you choose L; not 'too small',
    not 'too big'!

    """

    return 0. if np.abs(t) > L else .5 * (1 + scipy.cos(np.pi * t / L))


def symmetrized(x, flip=True):
    """Makes a 1D array symmetric about it's zeroth value.
    Useful for reflecting a function about the ordinate axis.

    Parameters
    ----------
    flip: boolean (optional, default True)
        if True, then the left half of the reflected array will
        be the reversal of -x, otherwise it'll be the reversal of
        x itself

    Returns
    -------
    Symmetrized array (of length 2 * len(x) - 1)

    """

    a = -1. if flip else 1.
    _x = np.hstack((np.flipud(a * x[1:]), x))

    return _x


def do_slicetiming(acquired_signal,
                   n_slices,
                   slice_index,
                   ref_slice=0,
                   user_time=None,
                   slice_order='ascending',
                   interleaved=False,
                   L=10,
                   symmetrization_trick=True,
                   ):
    """Function does sinc interpolation with Hanning window (faster,
    preserves frequency spectrum, and theoretically prevents artefacts
    to 'teleport' themselves accross TR's)

    XXX However, sinc interpolation is known to cause severe ringing due
    discontinuous edges in the input data sample

    Parameters
    ----------
    acquired_signal: array-like
        1D array of BOLD values (time-course from single voxel) or 2D array
        of BOLD values (time-courses from multiple voxels in thesame
        slice, one time-course per row) to be temporally interpolated

    n_slices: int
        number of slices per TR

    slice_index: int
        index of this slice in the bail of slices, according to the respective
        acquisition order

    ref_slice: int (optional, default 0)
        the slice number to be taken as the reference index

    slice_order: string or array of ints or length `n_slices`
        slice order of acquisitions in a TR
        'ascending': slices were acquired from bottommost to topmost
        'descending': slices were acquired from topmost to bottommost

    interleaved: bool (optional, default False)
        if set, then slices were acquired in interleaved order, odd-numbered
        slices first, and then even-numbered slices

    L: int (optional, default 10)
        width of Hanning Window to modify sinc kernel width, if this value
        if too large then no windowing will be applied (porous filter)

    symmetrization_trick: boolean (optional, default True)
        if set, the input signal will be reflected about ordinate axis to
        remove the inherent 'jump' in the input signal from one end-point
        of the 1D time grid the other

    Returns
    -------
    array: ST corrected signal, same shape as input `acquired_signal`

    """

    assert 0 <= slice_index < n_slices

    # dim sanity checks
    if not len(acquired_signal.shape) == 1:
        assert len(acquired_signal.shape) == 2
    n_scans = acquired_signal.shape[-1]

    ravel_output = False
    if len(acquired_signal.shape) == 1:
        ravel_output = True
        acquired_signal = acquired_signal.reshape(
            (1, len(acquired_signal)))

    n_voxels = acquired_signal.shape[0]

    # modify slice_index to be consistent with slice order
    if isinstance(slice_order, basestring):
        if interleaved:
            slices = range(n_slices)
            slice_index = (slices[1::2] + slices[0::2])[slice_index]
        if slice_order.lower() == 'ascending':
            slice_index = slice_index
        elif slice_order.lower() == 'descending':
            slice_index = n_slices - slice_index - 1
        else:
            raise TypeError("Unknown slice order '%s'!" % slice_order)
    else:
        # here, I'm assuming an explicitly specified slice order as a
        # permutation on n symbols
        assert len(slice_order) == n_slices
        slice_index = slice_order[slice_index]

    # compute shifting variables (TR normalized to 1)
    slice_TR = 1. / n_slices  # acq time for a single slice
    TA = n_scans - 1  # acq time of a all slices in a single 3D volume
    acquisition_time = np.linspace(0, TA, n_scans)  # acq times for this slice

    # sanitize user_time
    if user_time is None:
        # user didn't specify times they're interested in; will
        # just shift the acq times according to slice order / index
        shiftamount = (slice_index - ref_slice) * slice_TR
        user_time = acquisition_time - shiftamount

    # symmetrize the input signal
    if symmetrization_trick:
        acquisition_time = symmetrized(acquisition_time)
        acquired_signal = np.array([
                symmetrized(acquired_signal[j, :], flip=False)
                for j in xrange(n_voxels)])

    # compute sinc kernel
    time_deltas = np.array([
            user_time[j] - acquisition_time
            for j in xrange(len(user_time))])
    sinc_kernel = scipy.sinc(time_deltas)

    # modify the kernel width a Hanning window
    sinc_kernel *= np.vectorize(hanning_window)(
        time_deltas, L)

    # do interpolation proper
    print "[+] Reslicing slice %i/%i..." % (slice_index + 1, n_slices)
    st_corrected_signal = np.dot(
         acquired_signal,
         sinc_kernel.T,
         )

    # import smoothing_kernels as sk
    # if len(acquired_signal.shape) == 1:
    #     st_corrected_signal = sk.llreg(user_time, acquisition_time,
    #                                    acquired_signal)
    # else:
    #     st_corrected_signal = sk.llreg(user_time, acquisition_time,
    #                                    acquired_signal)

    # output shape must match input shape
    if ravel_output:
        st_corrected_signal = st_corrected_signal.ravel()

    return st_corrected_signal


def plot_slicetiming_results(TR,
                             acquired_signal,
                             st_corrected_signal,
                             ground_truth_signal=None,
                             ground_truth_time=None,
                             title="QA for Slice-Timing Correction",
                             ):

    """Function to generate QA plots post-STC business.

    Parameters
    ----------
    TR: float
        Repeation Time exploited by the STC algorithm

    acquired_signal: 1D array
        the input signal to the STC

    st_corrected_signal: array, same shape as `acquired_signal`
        the output corrected signal from the STC

    ground_truth_signal: 1D array (optional, default None), same length as
    `acquired_signal`
        ground truth signal

    ground_truth_time: array (optional, default None), same length as
    `ground_truth_time`
        ground truth time w.r.t. which the ground truth signal was collected

    """

    # sanity checks
    n_scans = len(acquired_signal)
    assert len(st_corrected_signal) == n_scans

    acquisition_time = np.linspace(0, (n_scans - 1) * TR, n_scans)

    N = None
    if ground_truth_signal is None:
        ground_truth_signal = st_corrected_signal
    else:
        N = len(ground_truth_signal)

    if ground_truth_time is None:
        assert len(ground_truth_signal) == len(acquisition_time)
        ground_truth_time = acquisition_time

    plt.figure()
    plt.xlabel('t')

    if not N is None:
        ax1 = plt.subplot2grid((3, 1), (0, 0),
                               rowspan=2)
    else:
        ax1 = plt.subplot2grid((3, 1), (0, 0),
                               rowspan=3)

    # plot true signal
    ax1.plot(ground_truth_time, ground_truth_signal, 'g')
    ax1.hold('on')

    # plot acquired signal (input to STC algorithm)
    ax1.plot(acquisition_time, acquired_signal, 'r--o')
    ax1.hold('on')

    # plot ST corrected signal
    ax1.plot(acquisition_time, st_corrected_signal, 'gs')
    ax1.hold('on')

    # misc
    if title:
        ax1.set_title(title)
    ax1.legend(('Ground-Truth signal',
               'Input sample',
               'Output ST corrected sample',
               ),
              loc='best')

    # plot error
    if not N is None:
        sampling_freq = (N - 1) / (n_scans - 1)  # XXX formula correct ??

        # acquire signal at same time points as corrected sample
        sampled_ground_truth_signal = ground_truth_signal[
            ::sampling_freq]

        # compute absolute error
        abs_error = np.abs(sampled_ground_truth_signal - st_corrected_signal)
        print 'SE:', (abs_error ** 2).sum()

        # plot abs error
        ax2 = plt.subplot2grid((3, 1), (2, 0),
                               rowspan=1)
        ax2.set_title(
            "Absolute Error (between ground-truth and corrected sample)")
        ax2.plot(acquisition_time, abs_error)

    plt.xlabel('time (s)')

    # show all generated plots
    pl.show()


def demo_HRF(n_slices=21,
             slice_index=-1,
             white_noise_std=1e-4):
    """STC for phase-shifted HRF in the presence of white-noise

    Parameters
    ----------
    n_slices: int (optional, default 21)
        number of slices per 3D volume of the acquisition

    slice_index: int (optiona, default -1)
        slice to which our voxel belongs

    white_noise_std: float (optional, default 1e-4)
        STD of white noise to add to phase-shifted sample (spatial corruption)

    """

    import math

    # sanity
    slice_index = slice_index % n_slices

    # create time values scaled at 1%
    timescale = .01
    n_timepoints = 24
    time = np.linspace(0, n_timepoints, num=1 + (n_timepoints - 0) / timescale)

    # create gamma functions
    n1 = 4
    lambda1 = 2
    n2 = 7
    lambda2 = 2
    a = .3
    c1 = 1
    c2 = .5

    def compute_hrf(t):
        """Auxiliary function to compute HRF at given times (t)

        """

        hx = (t ** (n1 - 1)) * np.exp(
            -t / lambda1) / ((lambda1 ** n1) * math.factorial(n1 - 1))
        hy = (t ** (n2 - 1)) * np.exp(
            -t / lambda2) / ((lambda2 ** n2) * math.factorial(n2 - 1))

        # create hrf = weighted difference of two gammas
        hrf = a * (c1 * hx - c2 * hy)

        return hrf

    # compute hrf
    signal = compute_hrf(time)

    # sample the time and the signal
    freq = 100
    TR = 3.
    sampled_time = time[::TR * freq]
    # sampled_signal = signal[::TR * freq]

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_index * slice_TR
    shifted_sampled_time = time_shift + sampled_time

    # acquire the signal at the corrupt sampled time points
    acquired_signal = compute_hrf(shifted_sampled_time,
                                  )

    # add white noise
    acquired_signal += white_noise_std * np.random.randn(
        len(acquired_signal))

    # do STC
    st_corrected_signal = do_slicetiming(
        acquired_signal, n_slices=n_slices,
        slice_index=slice_index,
        )

    # QA clinic
    plot_slicetiming_results(
        TR,
        acquired_signal,
        st_corrected_signal,
        ground_truth_signal=signal,
        ground_truth_time=time,
        title=("Slice-Timing Correction for sampled HRF time-course"
               " from a single voxel\nN.B:- TR = %.2f, # slices = %i, "
               "slice index = %i, white-noise std = %f"
               ) % (TR, n_slices, slice_index, white_noise_std)
        )


def demo_BOLD(dataset='spm_auditory',
              ):
    """XXX This only works on my machine since you surely don't have
    SPM single-subject auditory data or FSL FEEDS data installed on yours ;)

    XXX TODO: interpolation can produce signal out-side the brain;
    solve this with proper masking

    """

    # load fmri data
    import nibabel as ni
    import glob

    if dataset == 'spm_auditory':
        output_filename = "/tmp/st_corrected_spm_auditory.nii.gz"
        fmri_img = ni.concat_images(
            sorted(
                glob.glob(
                    ("/home/elvis/CODE/datasets/spm_auditory/fM00223"
                     "/fM00223_*.img")
                    )))
        fmri_data = fmri_img.get_data()[:, :, :, 0, :]

        TR = 7.
        slice_order = 'ascending'
    elif dataset == 'fsl-feeds':
        output_filename = "/tmp/st_corrected_fsl_feeds.nii.gz"
        fmri_img = ni.load(
            "/home/elvis/CODE/datasets/fsl-feeds-data/fmri.nii.gz")
        fmri_data = fmri_img.get_data()

        TR = 3.
        slice_order = 'ascending'
    elif dataset == 'localizer':
        output_filename = "/tmp/st_corrected_localizer.nii.gz"
        fmri_img = ni.load(
            "/home/elvis/.nipy/tests/data/s12069_swaloc1_corr.nii.gz")
        fmri_data = fmri_img.get_data()

        TR = 2.4
        slice_order = 'ascending'
    elif dataset == 'face_rep_SPM5':
        output_filename = "/tmp/st_corrected_spm_auditory.nii.gz"
        fmri_files = sorted(
            glob.glob(
                ("/home/elvis/CODE/datasets/face_rep_SPM5/RawEPI/"
                 "/sM03953_0005_*.img")
                ))
        fmri_img = ni.concat_images(fmri_files)

        TR = 2.
        slice_order = 'descending'
    else:
        raise RuntimeError("Unknown dataset: %s" % dataset)

    n_slices = fmri_data.shape[2]

    # sanity
    slice_index = np.arange(fmri_data.shape[2])

    # do full-brain STC
    corrected_fmri_data = np.array([do_slicetiming(
                fmri_data[:, :, z, :].reshape((
                        -1,
                         fmri_data.shape[-1])),
                n_slices=n_slices,
                slice_index=z,
                slice_order=slice_order,
                )
                                    for z in slice_index])

    # the output has shape n_slices x n_voxels_per_slice x n_scans
    # reshape it to the input's shape n_x x n_y x n_z x n_xcans
    corrected_fmri_data = corrected_fmri_data.swapaxes(0, 1).reshape(
        fmri_data.shape)

    # # XXX don't interpolate the beginning (problematique, unless we did zero
    # # padding in time (resp. freq domain) or a similar trick !!
    # w = 2
    # corrected_fmri_data[:, :, :, :w] = fmri_data[:, :, :, :w]

    # # XXX don't interpolate the beginning (problematique, unless we did zero
    # # padding in time (resp. freq domain) or a similar trick !!
    # corrected_fmri_data[:, :, :, -w:] = fmri_data[:, :, :, -w:]

    # save output unto disk
    ni.save(ni.Nifti1Image(corrected_fmri_data, fmri_img.get_affine()),
            output_filename)

    # print "\r\n[+] Wrote ST corrected image to %s\r\n" % output_filename

    # QA clinic
    for z in slice_index:
        x, y = 32, 32
        plot_slicetiming_results(
            TR,
            fmri_data[x, y, z, :],
            corrected_fmri_data[x, y, z, :],
            title=("Slice-Timing Correction BOLD time-course from a single"
                   "voxel \nN.B:- TR = %.2f, # slices = %i, x = %i, y = %i,"
                   " slice index (z) = %i") % (TR, n_slices, x, y, z)
            )


def my_sinusoid(time, frequency=1.):
    """Creates mixture of sinusoids with different frequencies

    """

    if not hasattr(frequency, '__len__'):
        frequency = [frequency]

    res = time * 0

    for f in frequency:
        res += np.sin(2 * np.pi * time * f)

    return res


def demo_sinusoid(n_slices=10, slice_index=-1, white_noise_std=1e-2):
    """STC for time phase-shifted sinusoidal mixture in the presence of
    white-noise. This is supposed to be a BOLD time-course from a single
    voxel.

    Parameters
    ----------
    n_slices: int (optional, default 10)
        number of slices per 3D volume of the acquisition

    slice_index: int (optiona, default -1)
        slice to which our voxel belongs

    white_noise_std: float (optional, default 1e-2)
        STD of white noise to add to phase-shifted sample (spatial corruption)

    """

    timescale = .01
    sine_freq = [.5, .8]  # number of complete cycles per unit time

    time = np.arange(0, 24 + timescale, timescale)
    signal = my_sinusoid(time,
                         frequency=sine_freq,
                         )

    # sanity
    slice_index = slice_index % n_slices

    # define timing vars
    freq = 10
    TR = freq * timescale

    # sample the time
    sampled_time = time[::freq]

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_index * slice_TR
    shifted_sampled_time = time_shift + sampled_time

    # acquire the signal at the corrupt sampled time points
    acquired_signal = my_sinusoid(shifted_sampled_time,
                                  frequency=sine_freq,
                                  )

    # add white noise
    acquired_signal += white_noise_std * np.random.randn(
        len(acquired_signal))

    # do STC
    st_corrected_signal = do_slicetiming(
        acquired_signal,
        n_slices=n_slices,
        slice_index=slice_index,
        )

    # QA clinic
    plot_slicetiming_results(
        TR,
        acquired_signal,
        st_corrected_signal,
        ground_truth_signal=signal,
        ground_truth_time=time,
        title=("Slice-Timing Correction for sampled sinusoidal BOLD "
               "time-course from a single voxel \nN.B:- TR = %.2f, # "
               "slices = %i, slice index = %i, white-noise std = %f"
               ) % (TR, n_slices, slice_index, white_noise_std)
        )

if __name__ == '__main__':
    # demo_BOLD(dataset='spm_auditory')
    demo_sinusoid(n_slices=100, slice_index=50)
    demo_HRF(n_slices=100, slice_index=50)
