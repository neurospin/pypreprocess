"""
:Module: slice_timing
:Synopsis: Module for STC
:Author: elvis[dot]dohmatob[at]inria[dot]fr

"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy

# some useful constants
INFINITY = np.inf


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


def symmetricized(x, flip=True):
    """Makes an array (1D or 2D) symmetric about it's zeroth value, and along
    the last axis. This is useful for reflecting a function about the
    ordinate axis.

    Parameters
    ----------
    flip: boolean (optional, default True)
        if True, then the left half of the reflected array will
        be the reversal of -x, otherwise it'll be the reversal of
        x itself

    Returns
    -------
    Symmetricized array (of length 2 * len(x) - 1)

    """

    a = -1. if flip else 1.

    if len(x.shape) == 1:
        _x = np.hstack((np.flipud(a * x[1:]), x))
    else:
        _x = np.array([np.hstack((np.flipud(a * x[j, 1:]), x[j, :] ))
                       for j in xrange(x.shape[0])])

    return _x


def get_slice(data, k):
    """Function retrieves kth slice of 4D brain data

    Returns
    -------
    2D array of shape (n_voxels, n_scans), where n_voxels is the
    product of the first 2 dimensions of data and n_scans is the
    last dimension of data

    """

    # sanity checks
    assert len(data.shape) == 4
    assert 0 <= k < data.shape[2]

    return data[:, :, k, :].reshape((
            -1,
             data.shape[-1]))


def get_acquisition_time(n_scans, TR=1.):
    """Function computes the acquisitions of complete 3D volumes
    in a 4D film, as multiples of the Repeatition Time (TR)

    n_scans: int
        number of scans (TRs) in the 4D film

    TR: float (optional, default 1)
        the Time of Repeatition

    Returns
    -------
    acquisition_times: array
        instants at which the respective 3D volumes
        where acquired, as multiples of the TR

    """

    # acq time of a all slices in a single 3D volume
    TA = (n_scans - 1) * TR

    # acq times for this slice
    acquisition_time = np.linspace(0, TA, n_scans)

    return acquisition_time


def apply_sinc_STC(data, sinc_kernel, symmetricize_data=False):
    """Apply a sinc STC tranform (aka a sinc kernel) to raw data

    """

    # symmetricize the data about the origin of the ordinate axis
    if symmetricize_data:
        data = symmetricized(data, flip=False)

    # apply the transform
    return np.dot(data,
                  sinc_kernel.T)


def fix_slice_index(slice_index, n_slices, slice_order='ascending',
                    interleaved=False,):
    """Function fixes a slice index (an interger in the range [0, n_slices))
    so that it is consistent with the specified slice order.

    Parameters
    ----------
    slice_index: int
        slice index to fix, integer in the range [0, n_slices)

    n_slices: int
        the number of slices there're altogether

    slice_order: string or array of ints or length `n_slices`
        slice order of acquisitions in a TR
        'ascending': slices were acquired from bottommost to topmost
        'descending': slices were acquired from topmost to bottommost

    interleaved: bool (optional, default False)
        if set, then slices were acquired in interleaved order, odd-numbered
        slices first, and then even-numbered slices

    Returns
    -------
    slice_index: int
        fixed slice index

    Raises
    ------
    TypeError

    """

    # sanity check
    assert 0 <= slice_index < n_slices

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

    return slice_index


def get_user_time(user_time, n_slices, n_scans, slice_index=None,
                  ref_slice=0):
    """Function to compute/sanitize user time (aka times at which
    user wants response values to be predicted (via temporal
    interpolation)

    Returns
    -------
    user_time: array of time instants at which response values
    will be predicted by subsequent STC

    Parameters
    ----------
    n_slices: int
        number of slices per TR

    n_scans: int
        number of scans (TRs) in the underlying experiment

    slice_index: int
        index of this slice in the bail of slices, according to the respective
        acquisition order

    ref_slice: int (optional, default 0)
        the slice number to be taken as the reference index

    user_time: string, float, or array of floats (optional, default "compute")
        times user what us to predict values at
        string "compute": user wants us to do standard STC, in which the slice
        at slice_index is shifted by an amount slice_index * TR / n_slices to
        the left, in time

        float: user wants us to predict the value at slice_TR + this shift.
        for example if this value is .5, then we'll predict the response values
        at time instants TR + .5TR, 2TR + .5TR, ..., (n_scans - 1)TR + .5TR;
        if the value if -.7TR, then the instants will be TR - .7TR, ...,,
        (n_scans - 1)TR - .7TR.
        N.B.:- This value must be in the range [0., 1.) .

        array of floats: response values for precisely this times will be
        predicted

    Raises
    ------
    TypeError

    """

    # compute shifting variables
    slice_TR = 1. / n_slices  # acq time for a single slice
    acquisition_time = get_acquisition_time(n_scans)

    if isinstance(user_time, basestring):
        if user_time == 'compute':
            if slice_index is None:
                raise ValueError(
                    ("A value of slice_index obligatory, since "
                     "you requested me to compute the user_time"))

            # user didn't specify times they're interested in; we'll
            # just shift the acq time to the left
            shiftamount = (slice_index - ref_slice) * slice_TR
            user_time = acquisition_time - shiftamount
        else:
            raise ValueError("Unknown user_time value: %s" % user_time)
    if isinstance(user_time, float) or isinstance(user_time, int):
        if not 0. <= user_time < 1.:
            raise ValueError(
                "Value must be between 0 and 1; %f given" % user_time)

        shiftamount = user_time - ref_slice * slice_TR
        user_time = acquisition_time - shiftamount
    elif isinstance(user_time, np.ndarray) or isinstance(user_time, list):
        user_time = np.array(user_time)
    else:
        raise ValueError(
            "Invalid value of user_time specified: %s" % user_time)

    # return the computed/sanitized user_time
    return user_time


def do_slicetiming(acquired_signal,
                   n_slices,
                   user_time='compute',
                   slice_order='ascending',
                   interleaved=False,
                   slice_index=None,
                   ref_slice=0,
                   acquisition_time='compute',
                   jobtype='estwrite',
                   L=INFINITY,
                   symmetricization_trick=True,
                   display_kernel=True,
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

    user_time: string, float, or array of floats (optional, default "compute")
        times user what us to predict values at
        string "compute": user wants us to do standard STC, in which the slice
        at slice_index is shifted by an amount slice_index * TR / n_slices to
        the left, in time

        float: user wants us to predict the value at slice_TR + this shift.
        for example if this value is .5, then we'll predict the response values
        at time instants TR + .5TR, 2TR + .5TR, ..., (n_scans - 1)TR + .5TR;
        if the value if -.7TR, then the instants will be TR - .7TR, ...,,
        (n_scans - 1)TR - .7TR.
        N.B.:- This value must be in the range [0., 1.) .

        array of floats: response values for precisely this times will be
        predicted

    slice_order: string or array of ints or length `n_slices`
        slice order of acquisitions in a TR
        'ascending': slices were acquired from bottommost to topmost
        'descending': slices were acquired from topmost to bottommost

    interleaved: bool (optional, default False)
        if set, then slices were acquired in interleaved order, odd-numbered
        slices first, and then even-numbered slices

    jobtype: string (optional, default "estwrite")
        if value if "estimate", then the computed sinc kernel (transform) will
        be returned as output, otherwise if its value is "estwrite", and the
        computed transform is applied to the imput signal, and the result of
        the tranformation (the resliced signal/data) is returned

    L: int (optional, default INFINITY)
        width of Hanning Window to modify sinc kernel width, if this value
        if too large then no windowing will be applied (porous filter)

    symmetricization_trick: boolean (optional, default True)
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

    # modify slice_index to be consistent with slice order
    slice_index = fix_slice_index(slice_index, n_slices,
                                  slice_order=slice_order,
                                  interleaved=interleaved,
                                  )

    # compute/sanitize acquision time
    if isinstance(acquisition_time, basestring):
        if acquisition_time == 'compute':
            acquisition_time = get_acquisition_time(n_scans)
        else:
            raise ValueError("Unknown acquisition_time value: %s" % user_time)
    elif isinstance(acquisition_time, np.ndarray) or isinstance(
        acquisition_time, list):
        acquisition_time = np.array(acquisition_time)
    else:
        raise ValueError(
            "Unknown acquisition_time type: %s" % type(acquisition_time))

    # get/sanitize user time
    user_time = get_user_time(user_time, n_slices, n_scans,
                              slice_index=slice_index, ref_slice=ref_slice,)

    # symmetricize the acq time
    if symmetricization_trick:
        acquisition_time = symmetricized(acquisition_time)

    # compute sinc kernel
    print "[+] Estimating STC tranform for slice %i/%i..." % (
        slice_index + 1, n_slices)
    time_deltas = np.array([
            user_time[j] - acquisition_time
            for j in xrange(len(user_time))])
    sinc_kernel = scipy.sinc(time_deltas)

    # modify the kernel width a Hanning window
    sinc_kernel *= np.vectorize(hanning_window)(
        time_deltas, L)

    # display sinc kernel
    if display_kernel:
        pl.plot(user_time, sinc_kernel)
        pl.xlabel("time (s)")
        pl.ylabel("kernel value")
        pl.title("%i Kernels (smoothed with Hanning Window of width = %.1f ) "
                 "around %i user time points" % (len(user_time), L,
                                                 len(user_time)))
        pl.show()

    # set output to sinc kernel
    output = sinc_kernel

    # apply the learnt tranform (sinc kernel indeed)
    if jobtype == 'estwrite':
        print "[+] Reslicing slice %i/%i..." % (slice_index + 1, n_slices)
        st_corrected_signal = apply_sinc_STC(
            acquired_signal,
            sinc_kernel,
            symmetricize_data=symmetricization_trick,
            )

        output = st_corrected_signal

    # sanitize output shape
    if ravel_output:
        output = output.ravel()

    # return to caller
    return output


class STC(object):
    def fit(self, raw_data, slice_order='ascending', interleaved=False,
            ref_slice=0, user_time='compute', symmetricization_trick=True,
            ):
        """Computes a transform for ST correction

        Parameters
        ----------
        raw_data: array
            4D array for input fMRI data

        ref_slice: int (optional, default 0)
            the slice number to be taken as the reference index

        user_time: string, float (optional, default "compute")
            times user what us to predict values at. Possible values are:

            string "compute": user wants us to do standard STC, in which
            the slice at slice_index is shifted by an amount slice_index *
            TR / n_slices to the left, in time

            float: user wants us to predict the value at slice_TR + this shift.
            for example if this value is .5, then we'll predict the response
            values at time instants TR + .5TR, 2TR + .5TR, ..., (n_scans - 1)TR
            + .5TR; if the value if -.7TR, then the instants will be TR - .7TR,
            ...,, (n_scans - 1)TR - .7TR.
            N.B.:- This value must be in the range [0., 1.) .

        Returns
        -------
        self._tranform: 3D array of shape (n_slices, n_user_time, n_scans) or
        (n_slices, n_user_time, 2n_scans - 1) if symmetricization_trick is set,
        where n_user_time is the number of time points user is requesting
        response prediction for

        """

        # sanity checks
        if len(raw_data.shape) == 4:
            self._n_slices = raw_data.shape[2]
        elif len(raw_data) == 3:
            self._n_slices = raw_data.shape[0]
        else:
            raise ValueError("Inpur raw data must be 3D or 4D array")

        # set meta params
        self._n_scans = raw_data.shape[-1]
        self._raw_data = raw_data
        self._slice_order = slice_order
        self._interleaved = interleaved
        self._user_time = user_time
        self._ref_slice = ref_slice
        self._symmetricization_trick = symmetricization_trick

        # compute acquisition time
        self._acquisition_time = get_acquisition_time(self._n_scans)

        # compute user time
        self._user_time = np.array([
                get_user_time(self._user_time, self._n_slices, self._n_scans,
                              slice_index=z, ref_slice=self._ref_slice,)
                for z in np.arange(self._n_slices)])

        # compute full brain ST correction transform
        self._transform = np.array([do_slicetiming(
                    get_slice(self._raw_data,
                              z),
                    n_slices=self._n_slices,
                    slice_index=z,
                    slice_order=self._slice_order,
                    interleaved=self._interleaved,
                    ref_slice=self._ref_slice,
                    acquisition_time=self._acquisition_time,
                    user_time=self._user_time[z, :],
                    jobtype='estimate',
                    symmetricization_trick=self._symmetricization_trick,
                    display_kernel=False,
                    )
                                    for z in np.arange(self._n_slices)])

        # return the calculated transform
        return self._transform

    def get_slice(self, z):
        if len(self._raw_data.shape) == 4:
            return get_slice(self._raw_data, z)
        else:
            return self._raw_data[z]

    def transform(self):
        """Applies the fitted transform to the input raw data

        Returns
        -------
        self._output_data: 3D array of shape (n_slices, n_voxels_per_slice,
        n_scans), the ST corrected data

        """

        # do full-brain ST correction
        self._output_data = np.array([
                apply_sinc_STC(get_slice(self._raw_data, z),
                               self._transform[z],
                               symmetricize_data=self._symmetricization_trick,
                               )
                for z in xrange(self._n_slices)])

        # the output has shape n_slices x n_voxels_per_slice x n_scans
        # reshape it to the input's shape (n_x, n_y, n_z, n_xcans)
        self._output_data = self._output_data.swapaxes(0, 1).reshape(
        self._raw_data.shape)

        # returned the transformed data
        return self._output_data


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
              QA=True,
              ):
    """XXX This only works on my machine since you surely don't have
    SPM single-subject auditory data or FSL FEEDS data installed on yours ;)

    XXX TODO: interpolation can produce signal out-side the brain;
    solve this with proper masking

    """

    # load fmri data
    import nibabel as ni
    import glob

    slice_order = 'ascending'
    interleaved = False
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
    elif dataset == 'fsl-feeds':
        output_filename = "/tmp/st_corrected_fsl_feeds.nii.gz"
        fmri_img = ni.load(
            "/home/elvis/CODE/datasets/fsl-feeds-data/fmri.nii.gz")
        fmri_data = fmri_img.get_data()

        TR = 3.
    elif dataset == 'localizer':
        output_filename = "/tmp/st_corrected_localizer.nii.gz"
        fmri_img = ni.load(
            "/home/elvis/.nipy/tests/data/s12069_swaloc1_corr.nii.gz")
        fmri_data = fmri_img.get_data()

        TR = 2.4
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
    slice_index = np.arange(n_slices)

    # fit STC
    stc = STC()
    print "[+] Estimating full-brain STC tranform.."
    stc.fit(fmri_data, slice_order=slice_order, interleaved=interleaved,)
    print "[+] Done."

    # do full-brain ST correction
    print "[+] Applying full-brain STC transform..."
    corrected_fmri_data = stc.transform()
    print "[+] Done."

    # save output unto disk
    ni.save(ni.Nifti1Image(corrected_fmri_data, fmri_img.get_affine()),
            output_filename)

    print "\r\n[+] Wrote ST corrected image to %s\r\n" % output_filename

    # QA clinic
    if QA:
        for z in slice_index:
            x, y = 32, 32
            plot_slicetiming_results(
                TR,
                fmri_data[x, y, z, :],
                corrected_fmri_data[x, y, z, :],
                title=(
                    "Slice-Timing Correction BOLD time-course from a single"
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
    sine_freq = [.5, .8, .11, .7]  # number of complete cycles per unit time

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
    demo_BOLD(dataset='fsl-feeds', QA=False)
    demo_sinusoid(n_slices=100, slice_index=50)
    demo_HRF(n_slices=100, slice_index=-1)
