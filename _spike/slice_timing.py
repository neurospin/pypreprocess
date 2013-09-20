"""
:Module: slice_timing
:Synopsis: Module for STC
:Author: elvis[dot]dohmatob[at]inria[dot]fr

"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import nibabel as ni

# some useful constants
INFINITY = np.inf


def plot_slicetiming_results(TR,
                             acquired_signal,
                             st_corrected_signal,
                             ground_truth_signal=None,
                             ground_truth_time=None,
                             title="QA for Slice-Timing Correction",
                             check_fft=True,
                             ):

    """Function to generate QA plots post-STC business, for a single voxel

    Parameters
    ----------
    TR: float
        Repeation Time exploited by the STC algorithm
    acquired_signal: 1D array
        the input signal to the STC
    st_corrected_signal: array, or list of arrays, same shape as
    acquired_signal
        the output corrected signal from the STC, or different STC
        implementations (one signal  per implelmentation)
    ground_truth_signal: 1D array (optional, default None), same length as
    `acquired_signal`
        ground truth signal
    ground_truth_time: array (optional, default None), same length as
    `ground_truth_time`
        ground truth time w.r.t. which the ground truth signal was collected

    """

    # sanity checks
    st_corrected_signal = np.array(st_corrected_signal)
    n_scans = len(acquired_signal)
    assert len(acquired_signal.shape) == 1
    assert st_corrected_signal.shape[-1] == n_scans

    if len(st_corrected_signal.shape) == 1:
        st_corrected_signal = np.array([st_corrected_signal])

    acquisition_time = np.linspace(0, (n_scans - 1) * TR, n_scans)

    if len(st_corrected_signal.shape) == 1:
        n_methods = 1
    else:
        n_methods = st_corrected_signal.shape[0]

    N = None
    if ground_truth_signal is None:
        if n_methods == 1:
            ground_truth_signal = st_corrected_signal
    else:
        N = len(ground_truth_signal)

    if ground_truth_time is None:
        if N:
            assert len(ground_truth_signal) == n_scans
            ground_truth_time = acquisition_time

    plt.rc('legend', fontsize=8,)

    if title:
        plt.suptitle(title, fontsize=14)

    n_rows = 1
    n_rows += int(check_fft) + int(not N is None)

    ax1 = plt.subplot2grid((n_rows, 1),
                      (0, 0))
    ax1_legends = []

    # plot ground-truth signal
    ax1.plot(ground_truth_time, ground_truth_signal)
    ax1_legends.append("Ground-truth signal")
    ax1.hold('on')

    # plot acquired sample
    ax1.plot(acquisition_time, acquired_signal, '--o')
    ax1_legends.append("Orignal sample")
    ax1.hold('on')

    if check_fft:
        ax2 = plt.subplot2grid((n_rows, 1),
                               (1, 0))
        ax2_legends = []

        # plot fft of acquired sample
        ax2.plot(acquisition_time[1:],
                 np.abs(np.fft.fft(acquired_signal))[1:])
        ax2_legends.append("Original sample")
        ax2.hold('on')

    if not N is None:
        sampling_freq = (N - 1) / (n_scans - 1)  # XXX formula correct ??

        # acquire signal at same time points as corrected sample
        sampled_ground_truth_signal = ground_truth_signal[
            ::sampling_freq]

        ax3 = plt.subplot2grid((n_rows, 1),
                           (2, 0))
        ax3_legends = []

    for j in xrange(n_methods):
        # plot ST corrected signal
        ax1.plot(acquisition_time, st_corrected_signal[j], 's-')
        ax1_legends.append("STC method %i" % (j + 1))
        ax1.hold('on')

        if check_fft:
            # plot fft of ST corrected sample
            ax2.plot(acquisition_time[1:],
                     np.abs(np.fft.fft(st_corrected_signal[j]))[1:])
            ax2_legends.append(
                "STC method %i" % (j + 1))
            ax2.hold('on')

    #     # legend
    #     ax.legend(('Ground-Truth signal',
    #                'Input sample',
    #                'Output ST corrected sample',
    #                ),
    #               loc='best',
    #               )

    #     # title
    #     if n_methods > 1:
    #         ax.set_title("method %i" % j, fontsize=8)

        # plot error
        if not N is None:
            # compute absolute error
            abs_error = np.array([
                    np.abs(
                        sampled_ground_truth_signal - st_corrected_signal[j])
                    for j in xrange(n_methods)])

            ax3.plot(acquisition_time, abs_error.T)
            ax3_legends.append("STC method %i" % (j + 1))

    #     # legend
    #     if n_methods > 1:
    #         ax.legend(tuple(["method %i" % j for j in xrange(n_methods)]))

    # misc
    plt.xlabel('time (s)')
    ax1.legend(tuple(ax1_legends))
    ax1.set_title("Data")
    if check_fft:
        ax2.legend(tuple(ax2_legends), ncol=2)
        ax2.set_title("Absolute value of FFT")
    if N:
        ax3.legend(tuple(ax3_legends))
        ax3.set_title("Absolute Error")

    # show all generated plots
    pl.show()


def hanning_window(t, L):
    """This function computes the Hanning window(s) of given width and
    centre(s).

    Parameters
    ----------
    t: scalar or nd array of scalars
        the centre(s) of the Hanning window(s)

    L: int
        the width of the Hanning window(s)

    Notes
    -----
    Since sinc(t) decays to zero as t departs from zero, in the
    sinc interpolation sum for the BOLD signal value in a voxel at time
    instant, terms corresponding to acquisition times beyond a radius L
    of t can be dropped (their contribution to the said sum will be small).
    One way to do this is to convolve the sum with a Hanning window of
    appropriate width about t0. Mind how you choose L!

    Examples
    --------
    >>> import slice_timing as st
    >>> from numpy import *
    >>> a = linspace(-10,10, 5)
    >>> hw = st.hanning_window(a, 10)

    """

    # sanitize window width
    assert L > 0

    # compute and return the window
    return (np.abs(t) <= L) * .5 * (1 + scipy.cos(np.pi * t / L))


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

    Examples
    --------
    >>> import slice_timing as st
    >>> from numpy import *
    >>> a = linspace(0,10, 50)
    >>> symm_a = st.symmetricized(a)

    """

    a = -1. if flip else 1.

    if len(x.shape) == 1:
        _x = np.hstack((np.flipud(a * x[1:]), x))
    else:
        _x = np.array([np.hstack((np.flipud(a * x[j, 1:]), x[j, :] ))
                       for j in xrange(x.shape[0])])

    return _x


def get_acquisition_time(n_scans, TR=1.):
    """Function computes the acquisitions of complete 3D volumes
    in a 4D film, as multiples of the Repetition Time (TR)

    Parameters
    ----------
    n_scans: int
        number of scans (TRs) in the 4D film
    TR: float (optional, default 1)
        the Time of Repetition

    Returns
    -------
    acquisition_times: array
        instants at which the respective 3D volumes
        where acquired, as multiples of the TR

    Examples
    --------
    >>> import slice_timing as st
    >>> from numpy import *
    >>> a = linspace(-10,10, 100)
    >>> n_scans = 200
    >>> at = st.get_acquisition_time(n_scans)

    """

    # acq time of a all slices in a single 3D volume
    TA = (n_scans - 1) * TR

    # acq times for this slice
    acquisition_time = np.linspace(0, TA, n_scans)

    return acquisition_time


def apply_sinc_STC(data, sinc_kernel, symmetricize_data=False):
    """Apply a sinc STC transform (aka a sinc kernel) to raw data

    """

    # symmetricize the data about the origin of the ordinate axis
    if symmetricize_data:
        data = symmetricized(data, flip=False)

    # apply the transform (exploiting sparse structure)
    return scipy.sparse.csr_matrix(sinc_kernel).dot(data.T).T


def get_slice_indices(n_slices, slice_order='ascending',
                    interleaved=False,):
    """Function computes the slice indices, consistent with the
    specified slice order.

    Parameters
    ----------
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
    slice_indices: int
        slice indices consistent with slice order (i.e, slice_indices[k]
        if the corrected index of slice k according to the slice order)

    Raises
    ------
    Exception

    Examples
    --------
    >>> import slice_timing as st
    >>> from numpy import *
    >>> slice_indices = st.get_slice_indices(150)
    >>> slice_indices = st.get_slice_indices(150, slice_order='descending',
    ... interleaved=True)

    """

    # sanity check
    if isinstance(slice_order, basestring):
        slice_indices = range(n_slices)
        if interleaved:
            slice_indices = slice_indices[1::2] + slice_indices[0::2]
        if slice_order.lower() == 'ascending':
            pass
        elif slice_order.lower() == 'descending':
            slice_indices = np.flipud(slice_indices)
        else:
            raise Exception("Unknown slice order '%s'!" % slice_order)
    else:
        # here, I'm assuming an explicitly specified slice order as a
        # permutation on n symbols
        assert len(slice_order) == n_slices
        slice_order = np.array(slice_order, dtype=int)
        slice_indices = slice_order

    return slice_indices


def get_user_time(n_slices, n_scans, slice_index=None,
                  ref_slice=0, user_time='compute'):
    """Function to compute/sanitize user time (aka times at which
    user wants response values to be predicted (via temporal
    interpolation)

    Parameters
    ----------
    n_slices: int
        number of slices per TR
    n_scans: int
        number of scans (TRs) in the underlying experiment
    slice_index: int or array-like or ints (optional, default None)
        index(ces) or slice(s) for which we want to compute the user time.
        If this is None, then user times for every slice will be computed
    ref_slice: int (optional, default 0)
        the slice number to be taken as the reference index
    user_time: string, scalar, or array of scalars (optional, default
    "compute")
        times user what us to predict values at
        string "compute": user wants us to do standard STC, in which the slice
        at slice_index is shifted by an amount slice_index * TR / n_slices to
        the left, in time
        scalar: user wants us to predict the value at slice_TR + this shift.
        for example if this value is .5, then we'll predict the response values
        at time instants TR + .5TR, 2TR + .5TR, ..., (n_scans - 1)TR + .5TR;
        if the value if -.7TR, then the instants will be TR - .7TR, ...,,
        (n_scans - 1)TR - .7TR. In this case, the supplied value must be in
        the range [0., 1.)
        array of scalars: response values for precisely this times will be
        predicted

    Returns
    -------
    user_time: 1D array of size n_slices if slice_index is an int, or 2D
    array of shape (len(slice_index), n_slices) othewise (i.e if list of ints)

    Raises
    ------
    Exception

    Examples
    --------
    >>> import slice_timing as st
    >>> from numpy import *
    >>> ut = st.get_user_time(21, 150)

    """

    # sanitize slice_index
    if hasattr(slice_index, '__len__'):
        slice_index = np.array(slice_index)

    # compute shifting variables (WLOG, TR has been normalized to 1)
    slice_TR = 1. / n_slices  # acq time for a single slice
    acquisition_time = get_acquisition_time(n_scans)

    if isinstance(user_time, basestring):
        if user_time == 'compute':
            if slice_index is None:
                slice_index = np.arange(n_slices)
            # user didn't specify times they're interested in; we'll
            # just shift the acq time to the left
            shiftamount = (slice_index - ref_slice) * slice_TR
            if hasattr(slice_index, '__len__'):
                user_time = np.array([acquisition_time - delta
                                      for delta in shiftamount])
            else:
                user_time = acquisition_time - shiftamount
        else:
            raise Exception("Unknown user_time value: %s" % user_time)
    if isinstance(user_time, float) or isinstance(user_time, int):
        if not 0. <= user_time < 1.:
            raise Exception(
                "Value must be between 0 and 1; %f given" % user_time)

        shiftamount = user_time - ref_slice * slice_TR
        user_time = acquisition_time - shiftamount
    elif isinstance(user_time, np.ndarray) or isinstance(user_time, list):
        user_time = np.array(user_time)
    else:
        raise Exception(
            "Invalid value of user_time specified: %s" % user_time)

    # return the computed/sanitized user_time
    return user_time


def compute_sinc_kernel(acquisition_time, user_time,
                        symmetricization_trick=True,
                        L=None,
                        slice_index=None,
                        n_slices=None,
                        ):
    """Computes the sinc kernel around given user times (user_time).

    Parameters
    ----------
    acquisition_time: 1D array of size n_scans
        acquisition times for the TRs in the underlying experiment
        (typically 0, TR, 2TR, ..., (n_scans - 1) TR or 0, 1, 2, ...,
        (n_scans - 1)
    user_time: 1D array of shape n_user_times
        the times around which the kernels will be centered (this
        is the times your want to predict response for)
    symmetricization_trick: boolean (optional, default True)
        if true symmetricization trick will be used to reflect the
        acquisition times about the ordinate axis (this helps the
        subsequenc sinc-based interpolation)
    L: int (optional, default None)
        width of Hanning Window to use in windowing the sinc kernel
        (this should help preventing the 'teleportation' of artefacts across
        different TRs, and also make the kernel sparse, thus speeding up
        the ensuing linear algebra)

    Returns
    -------
    sinc_kernel: 2D array of shape (len(user_time), 2n_scans - 1) if
    symmetricization trick has been used or (len(user_time), n_scans)
    otherwise

    Raises
    ------
    AssertionError

    Examples
    --------
    >>> import slice_timing as st
    >>> from numpy import *
    >>> at = st.get_acquisition_time(10)
    >>> ut = st.get_user_time(21, 10, slice_index=9)
    >>> k = st.compute_sinc_kernel(at, ut)

    """

    # sanitize the times
    assert len(user_time.shape) == 1
    assert len(acquisition_time.shape) == 1

    # brag
    if not slice_index is None:
        if not n_slices is None:
            print ("Estimating STC transform (sinc kernel) for slice "
                   "%i/%i...") % (slice_index + 1, n_slices)
        else:
            print ("Estimating STC transform (sinc kernel) for slice "
                   "%i") % (slice_index + 1)

    # symmetricize the acq time
    if symmetricization_trick:
        acquisition_time = symmetricized(acquisition_time)

    # compute time shifts
    time_shift = np.array([
            t - acquisition_time
            for t in user_time])

    # compute kernel
    sinc_kernel = scipy.sinc(time_shift)

    # modify the kernel with a Hanning window of width L
    # around the user times (user_time)
    if not L is None and L != INFINITY:
        assert L > 0
        sinc_kernel *= hanning_window(
            time_shift, L)

    # return computed kernel
    return scipy.sparse.csr_matrix(sinc_kernel)


class STC(object):
    """Slice-Timing Correction: Class implements fit/transform API used in
    sklearn, nilearn, etc.

    My formulation of STC using Whittaker-Shannon (aka sinc, see Friston
    et al.'s paper for details) interpolation is linear algebraic, and
    renders itself well for vectorization.

    Indeed, the correction is done as follows (a similar argument applies
    in case the "symmetricization trick" is employed, except S_k now has
    about 2 times more rows...):

    Given acquistions in which the reference slice is k_0, the BOLD signal
    from slice k (i.e a 2D array of shape n_voxels_per_slice-by-n_scans) is
    corrected by post multiplying it with the transpose of S_k, where S_k
    is 2D array of shape n_scan-by-n_scan defined by

        S_k[i, j] = sinc(j - (k - k_0) / n_slices - i) ... (1),

    where the sinc function above is typically windowed with an auxiliary
    kernel (we use Hanning windows) of appropriate width.

    The interpolation formula is then

        corrected_B_k = dot(B_k, transpose(S_k)) ... (2)

    Notes
    -----
    1- In formula (1) above, it's assumed that k and k_0 have been mapped to
    their real values consistently with the underlying slice order /
    acquisition type.

    2- The factor 1 / n_slices that appears in the argument of sinc in formula
    (1) is the time of acquisiton of a single slice in a 3D volume as a
    multiple of the TR (Refresher: TR or Repetition time is the time taken to
    acquire a full 3D image of the brain, and is experiment specific);
    j - (k - k_0) / n_slices corresponds to the time that the reference slice
    k_0 was acquired in the jth TR relative the kth slice. Noting that the
    term i in the same formula can be rewritten as

        i = i - (k_0 - k_0) / n_slices,

    it should be clear to the reader that, the argument of the sinc function
    in formula (1) is nothing other than the difference between the ith
    acquistion of the reference slice during the ith TR and the jth
    acquisition of the kth slice during the jth TR.

    3- For each voxel v in the kth slice, formula (2) can be expanded as a
    (rather clumsy) sum of n_scan terms, which corresponds precisely to
    the vector-matrix product of the vth row of B_k and the transpose of S_k,
    the famous Whittaker-Shannon interpolation formula.

    4- The sinc can be replaced by any other appropriate kernel; (2) is just
    a convolution sum.

    Open Problem
    ------------
    why not 'learn' the 3D array S_ (the per slice kernels), with some
    reasonable axioms on it (e.g, it should leave the data invariant in the
    limit TR -> 0+, it should preserve the frequency spectra of the per voxel
    input BOLD signals, it should be sparse, blablabla ...) ?

    -
    3Lv15

    Examples
    --------
    >>> import slice_timing as st
    >>> from numpy import *
    >>> nx = 64
    >>> ny = 80
    >>> nz = 21
    >>> n_slices = nz
    >>> n_scans = 96
    >>> brain_shape = (nx, ny, n_slices, n_scans)
    >>> brain_data = random.random(brain_shape)
    >>> stc = st.STC()
    >>> kernels = stc.fit(raw_data=brain_data)
    Estimating STC transform (sinc kernel) for slice 1/21...
    Estimating STC transform (sinc kernel) for slice 2/21...
    Estimating STC transform (sinc kernel) for slice 3/21...
    Estimating STC transform (sinc kernel) for slice 4/21...
    Estimating STC transform (sinc kernel) for slice 5/21...
    Estimating STC transform (sinc kernel) for slice 6/21...
    Estimating STC transform (sinc kernel) for slice 7/21...
    Estimating STC transform (sinc kernel) for slice 8/21...
    Estimating STC transform (sinc kernel) for slice 9/21...
    Estimating STC transform (sinc kernel) for slice 10/21...
    Estimating STC transform (sinc kernel) for slice 11/21...
    Estimating STC transform (sinc kernel) for slice 12/21...
    Estimating STC transform (sinc kernel) for slice 13/21...
    Estimating STC transform (sinc kernel) for slice 14/21...
    Estimating STC transform (sinc kernel) for slice 15/21...
    Estimating STC transform (sinc kernel) for slice 16/21...
    Estimating STC transform (sinc kernel) for slice 17/21...
    Estimating STC transform (sinc kernel) for slice 18/21...
    Estimating STC transform (sinc kernel) for slice 19/21...
    Estimating STC transform (sinc kernel) for slice 20/21...
    Estimating STC transform (sinc kernel) for slice 21/21...

    >>> resliced_brain_data = stc.transform()

    """

    def __init__(self):
        """Default constructor

        """

        self._n_scans = None
        self._n_slices = None
        self._raw_data = None
        self._output_data = None
        self._transform = None

    def _check_raw_data_dims(self, raw_data):
        """Does sanity checks on the shape of raw_data

        Raises
        ------
        Exception

        """

        assert not raw_data is None

        if len(raw_data.shape) == 4:
            n_slices = raw_data.shape[2]
        elif len(raw_data.shape) == 3:
            n_slices = raw_data.shape[0]
        else:
            raise Exception(
                ("Input raw data must be 3D or 4D array, got"
                 " %s") % str(raw_data.shape))

        n_scans = raw_data.shape[-1]

        if n_slices != self._n_slices:
            raise Exception(
                ("raw_data is wrong shape (expected %i slices)"
                 ) % self._n_slices)
        if n_scans != self._n_scans:
            raise Exception(
                ("raw_data is wrong shape (expected %i scans/volumes)"
                 ) % self._n_scans)

    def _set_n_scans(self, n_scans):
        """Sets the value of the _n_scans field, after doing some sanity checks
        on the specified value

        Raises
        ------
        Exception

        """

        if (not isinstance(n_scans, int)) or n_scans < 1:
            raise Exception(
                ("n_scans argument must be a an integer > 1, got %s"
                 ) % n_scans)
        else:
            self._n_scans = n_scans

    def _set_n_slices(self, n_slices):
        """Sets the value of the _n_slices field, after doing some sanity
        checks on the specified value

        Raises
        ------
        Exception

        """

        if (not isinstance(n_slices, int)) or n_slices < 1:
            raise Exception(
                ("n_slices argument must be a an integer > 1, got %s"
                 ) % n_slices)
        else:
            self._n_slices = n_slices

    def _load_raw_data(self, raw_data):
        if isinstance(raw_data, basestring):
            return ni.load(raw_data).get_data()
        elif isinstance(raw_data, ni.Nifti1Image):
            return raw_data.get_data()
        else:
            return raw_data

    def _set_raw_data(self, raw_data):
        """Sets the value of the _raw_data field, after doing some sanity
        checks on the specified value

        Raises
        ------
        Exception

        """

        if not raw_data is None:
            self._load_raw_data(raw_data)
            self._check_raw_data_dims(raw_data)
            self._raw_data = raw_data

    def fit(self, n_scans=None, n_slices=None, raw_data=None,
            slice_order='ascending', interleaved=False,
            ref_slice=0,
            user_time='compute',
            L=None,
            symmetricization_trick=True,
            ):
        """Computes a transform for ST correction. The computed transform
        is not applied to data right away; this action is postponed until
        you explicitly invoke the transform(..) method.

        Parameters
        ----------
        raw_data: 4D array of shape (n_x, n_y, n_slices, n_scans) or 3D
        array of shape (n_slices, n_voxels_per_slice, n_scans) (optional,
        default None), `nibabel.Nifti1Image`, or string (filename)
            input data for STC (transforms will be fitted against this)
        ref_slice: int (optional, default 0)
            the slice number to be taken as the reference slice
        user_time: string, float (optional, default "compute")
            times user wants us to predict values at. Possible values are:
            string "compute": user wants us to do standard STC, in which
            the slice at index slice_index is shifted by an amount
            slice_index * TR / n_slices to the left, in time
            float: user wants us to predict the values at jth TR +
            user_time * TR for example if this value is .5, then we'll
            predict the response values at time instants TR + .5TR, 2TR +
            .5TR, ..., (n_scans - 1)TR + .5TR; if the value if -.7TR, then
            the instants will be TR - .7TR, ..., (n_scans - 1)TR - .7TR.
            N.B.:- This value must be in the range [0., 1.) .
        L: int (optional, default 50)
            width of Hanning Window to use in windowing the sinc kernel
            (this should help preventing the 'teleportation' of artefacts
            across different TRs, and also make the kernel sparse, thus
            speeding up the ensuing linear algebra)
        symmetricization_trick: boolean (optional, default True)
            if true symmetricization trick will be used to reflect the
            acquisition times about the ordinate axis (this helps the
            subsequenc sinc-based interpolation)

        Returns
        -------
        self._transform: 3D array of shape (n_slices, n_user_time, n_scans) or
        (n_slices, n_user_time, 2n_scans - 1) if symmetricization_trick is set,
        where n_user_time is the number of time points user is requesting
        response prediction for. This transform is applied to the input data
        when you later invoke the transform method.

        Raises
        ------
        Exception

        """

        # sanity checks on raw_data, n_scans, and n_slices
        raw_data = self._load_raw_data(raw_data)

        self._n_scans = n_scans
        self._n_slices = n_slices
        if not raw_data is None:
            if len(raw_data.shape) == 4:
                self._set_n_slices(raw_data.shape[2])
            elif len(raw_data.shape) == 3:
                self._set_n_slices(raw_data.shape[0])
            else:
                raise Exception(
                    ("Input raw data must be 3D or 4D array, got"
                     " %s") % str(raw_data.shape))

            self._set_n_scans(raw_data.shape[-1])
        else:
            if n_scans is None:
                raise Exception(
                    "raw_data not given, you must specify n_scans")
            self._set_n_scans(n_scans)
            if n_slices is None:
                raise Exception(
                    "raw_data not given, you must specify n_slices")
            else:
                self._set_n_slices(n_slices)

        self._set_raw_data(raw_data)

        # set other meta params
        self._slice_order = slice_order
        self._interleaved = interleaved
        self._user_time = user_time
        self._ref_slice = ref_slice
        self._L = L
        self._symmetricization_trick = symmetricization_trick

        # get slice indices, consistently with the slice order
        self._slice_indices = get_slice_indices(self._n_slices,
                                                slice_order=self._slice_order,
                                                interleaved=self._interleaved,
                                                )

        # fix ref slice index, to be consistent with the slice order
        # (acquisition type)
        self._ref_slice = self._slice_indices[self._ref_slice]

        # compute acquisition times
        self._acquisition_time = get_acquisition_time(self._n_scans)

        # compute user times (times for which the BOLD signal values
        # will be predicted for all voxels)
        self._user_time = get_user_time(self._n_slices,
                                        self._n_scans,
                                        slice_index=self._slice_indices,
                                        ref_slice=self._ref_slice,
                                        user_time=self._user_time,
                                        )

        # compute full brain ST correction transform
        self._transform = np.array([
                compute_sinc_kernel(
                    self._acquisition_time,
                    self._user_time[z],
                    symmetricization_trick=self._symmetricization_trick,
                    L=self._L,
                    slice_index=z,
                    n_slices=self._n_slices,
                    ) for z in self._slice_indices]
                                   )

        # return the calculated transform
        return self._transform

    def get_slice_data(self, z, raw_data=None):
        """Retrieves the raw data for a given slice.
        If raw_data is not specified, then the _raw_data field
        of this object is used in place.

        Parameters
        ----------
        z: int
            the slice index

        raw_data: 3D or 4D array (optional, default None)
            data whose zth slice is sought-for

        Returns
        -------
        2D array of shape (n_voxels_per_slice, n_scans)

        Raises
        ------
        Exception

        """

        # sanity checks on raw_data
        if raw_data is None:
            if self._raw_data is None:
                raise Exception(
                    ("raw_data not set; you must call this method with a "
                     "value for raw_data"))

            raw_data = self._raw_data
        if raw_data is None:
            raise Exception("raw_data field not set!")

        assert 0 <= z < self._n_slices

        if len(raw_data.shape) == 4:
            slice_data = raw_data[:, :, z, :]

            # check shape of slice_data
            if slice_data.shape[-1] != self._n_scans:
                raise Exception("Slice %i of raw_data is mal-shaped!" % z)

        elif len(raw_data.shape) == 3:
            slice_data = raw_data[z]

            # check shape of slice_data
            if slice_data.shape[-1] != self._n_scans:
                raise Exception("Slice %i of raw_data is mal-shaped!" % z)
        else:
            raise Exception("Mal-shaped raw_data; must be 3D or 4D array")

        # ravel slice_data to 2D array
        slice_data = slice_data.reshape((
                -1,
                 self._n_scans))

        return slice_data

    def get_slice_transorm(self, z):
        """Method returns the transform (sinc kernel) for the specified slice

        Parameters
        ----------
        z: int
            slice index for requested transform

        Returns
        -------
        2D array (compressed sparse)

        """

        assert 0 <= z < self._n_slices

        return self._transform[self._slice_indices[z]]

    def transform(self, raw_data=None,):
        """Applies the fitted transform to the input raw data. If raw_data
        is not specified, the _raw_data field of this object is used (probably
        supplied during call to fit(..) method.

        Parameters
        ----------
        raw_data: 4D array of shape (n_x, n_y, n_slices, n_scans) or 3D
        array of shape (n_slices, n_voxels_per_slice, n_scans) (optional,
        default None), `nibabel.Nifti1Image`, or string (filename)
            data to reslice

        Returns
        -------
        output_data: 3D array of shape (n_slices, n_voxels_per_slice,
        n_scans), the ST corrected data. This value can later be retrieved
        by invoking the get_last_output_data(..) method.

        Raises
        ------
        Exception

        """

        # sanity checks
        if raw_data is None:
            raw_data = self._raw_data
        if raw_data is None:
            raise Exception(
                ("raw_data not set during fitting, you must now call "
                 "transform(..) method with a value for the raw_data "
                 "argument"))
        self._check_raw_data_dims(raw_data)

        # do full-brain ST correction
        self._output_data = np.array([
                apply_sinc_STC(
                    self.get_slice_data(j, raw_data),
                    self.get_slice_transorm(j),
                    symmetricize_data=self._symmetricization_trick,
                    )
                for j in xrange(self._n_slices)])

        # sanitize output shape
        if len(raw_data.shape) == 4:
            # the output has shape (n_slices, n_voxels_per_slice, n_scans)
            # unravel it to match the input data's shape (n_x, n_y, n_slices,
            # n_xcans)
            self._output_data = self._output_data.swapaxes(0, 1).reshape(
                raw_data.shape)

        # return the transformed data
        return self._output_data

    def get_last_output_data(self):
        """Returns the output data computed by the last call to the transform
        method

        """

        if self._output_data is None:
            raise Exception(
                "You must first call transform(..) method to get output data!")

        return self._output_data

    def get_output_data(self):
        """Wrapper for get_last_output_data method

        """

        return self.get_last_output_data()

    def show_slice_transform(self, z):
        """Plots the transform a given slice index

        Parameters
        ----------
        z: int
            the slice index

        """

        n_user_time = self._user_time.shape[1]
        pl.plot(self._user_time[z], self._transform[z].toarray())
        pl.xlabel("time (s)")
        pl.ylabel("kernel value")
        pl.title("slice %i/%i : %i kernels around %i user times" % (
                z + 1, self._n_slices, n_user_time, n_user_time))

        pl.show()


def demo_HRF(n_slices=10,
             n_voxels_per_slice=1,
             white_noise_std=1e-4,
             ):
    """STC for phase-shifted HRF in the presence of white-noise

    Parameters
    ----------
    n_slices: int (optional, default 21)
        number of slices per 3D volume of the acquisition
    n_voxels_per_slice: int (optional, default 1)
        the number of voxels per slice in the simulated brain
        (setting this to 1 is sufficient for most QA since STC works
        slice-wise, and not voxel-wise)
    white_noise_std: float (optional, default 1e-4)
        STD of white noise to add to phase-shifted sample (spatial corruption)

    """

    print "\r\n\t\t ---demo_HRF---"

    import math

    slice_indices = np.arange(n_slices, dtype=int)

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
    n_scans = len(sampled_time)

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_indices * slice_TR
    shifted_sampled_time = np.array([tau + sampled_time
                                     for tau in time_shift])

    # acquire the signal at the corrupt sampled time points
    acquired_signal = np.array([
            [np.vectorize(compute_hrf)(shifted_sampled_time[j])
             for vox in xrange(n_voxels_per_slice)]
            for j in xrange(n_slices)])

    # add white noise
    acquired_signal += white_noise_std * np.random.randn(
        *acquired_signal.shape)

    # fit STC
    stc = STC()
    stc.fit(n_scans=n_scans, n_slices=n_slices)

    # apply STC
    print "Applying full-brain STC transform..."
    st_corrected_signal = stc.transform(acquired_signal)
    print "Done."

    # QA clinic
    print "Starting QA clinic (free entrance to the masses)..."
    for slice_index in xrange(n_slices):
        for vox in xrange(n_voxels_per_slice):
            plot_slicetiming_results(
                TR,
                acquired_signal[slice_index][0],
                st_corrected_signal[slice_index][0],
                ground_truth_signal=signal,
                ground_truth_time=time,
                title=(
                    "Slice-Timing Correction of sampled HRF time-course"
                    " from voxel %i of slice %i \nN.B:- TR = %.2f, "
                    "# slices = %i, # voxels per slice = %i, white-noise"
                    " std = %f") % (vox, slice_index, TR, n_slices,
                                    n_voxels_per_slice, white_noise_std,)
                )


def STC_QA(raw_fmri, corrected_fmri, TR, x, y, slice_indices=None,
           compare_with=None):

    assert raw_fmri.shape == corrected_fmri.shape

    n_slices = raw_fmri.shape[2]

    if slice_indices is None:
        slice_indices = np.arange(n_slices)

    assert np.all((0 <= slice_indices) & (slice_indices < n_slices))

    for z in slice_indices:
        output = corrected_fmri[x, y, z, :]
        if not compare_with is None:
            output = np.array([output, compare_with[x, y, z, :]])
        plot_slicetiming_results(
            TR,
            raw_fmri[x, y, z, :],
            output,
            title=(
                "Slice-Timing Correction of BOLD time-course from a single"
                " voxel \nN.B:- TR = %.2f, # slices = %i, x = %i, y = %i,"
                " slice index (z) = %i") % (TR, n_slices, x, y, z)
            )


def demo_BOLD(dataset='spm-auditory',
              QA=True,
              data_dir='/tmp/stc_demo',
              output_dir='/tmp',
              compare_with=None,
              ):
    """XXX This only works on my machine since you surely don't have
    SPM single-subject auditory data or FSL FEEDS data installed on yours ;)

    XXX TODO: interpolation can produce signal out-side the brain;
    solve this with proper masking

    Raises
    ------
    Exception

    """

    # sanitize dataset name
    assert isinstance(dataset, basestring)
    dataset = dataset.lower()

    # sanitize output dir
    if output_dir is None:
        output_dir = '/tmp'

    # demo specific imports
    import nibabel as ni
    import os
    import sys

    # load the data
    slice_order = 'ascending'
    interleaved = False
    if dataset == 'spm-auditory':
        # pypreproces path
        PYPREPROCESS_DIR = os.path.dirname(os.path.split(
                os.path.abspath(__file__))[0])
        sys.path.append(PYPREPROCESS_DIR)
        from datasets_extras import fetch_spm_auditory_data

        _subject_data = fetch_spm_auditory_data(data_dir)

        fmri_img = ni.concat_images(_subject_data['func'],)
        fmri_data = fmri_img.get_data()[:, :, :, 0, :]

        compare_with = ni.concat_images(
            [os.path.join(os.path.dirname(x),
                          "a" + os.path.basename(x))
             for x in _subject_data['func']]).get_data()

        TR = 7.
    elif dataset == 'fsl-feeds':
        PYPREPROCESS_DIR = os.path.dirname(os.path.split(
                os.path.abspath(__file__))[0])

        sys.path.append(PYPREPROCESS_DIR)
        from datasets_extras import fetch_fsl_feeds_data

        _subject_data = fetch_fsl_feeds_data(data_dir)
        if not _subject_data['func'].endswith('.gz'):
            _subject_data['func'] += '.gz'

        fmri_img = ni.load(_subject_data['func'],)
        fmri_data = fmri_img.get_data()

        TR = 3.
    elif dataset == 'localizer':
        output_filename = "/tmp/st_corrected_localizer.nii.gz"
        fmri_img = ni.load(
            "/home/elvis/.nipy/tests/data/s12069_swaloc1_corr.nii.gz")
        fmri_data = fmri_img.get_data()

        TR = 2.4
    elif dataset == 'face-rep-spm5':
        # XXX nibabel says the affines of the 3Ds are different
        fmri_img = ni.load(
            "/home/elvis/CODE/datasets/face_rep_SPM5/RawEPI/4D.nii.gz")
        fmri_data = fmri_img.get_data()

        TR = 2.
        slice_order = 'descending'
    else:
        raise Exception("Unknown dataset: %s" % dataset)

    output_filename = os.path.join(
        output_dir,
        "st_corrected_" + dataset.rstrip(" ").replace("-", "_") + ".nii.gz",
        )

    print "\r\n\t\t ---demo_BOLD (%s)---" % dataset

    # fit STC
    stc = STC()
    stc.fit(raw_data=fmri_data, slice_order=slice_order,
            interleaved=interleaved,
            )

    # do full-brain ST correction
    print "Applying full-brain STC transform..."
    corrected_fmri_data = stc.transform()
    print "Done."

    # save output unto disk
    print "Saving ST corrected image to %s..." % output_filename
    ni.save(ni.Nifti1Image(corrected_fmri_data, fmri_img.get_affine()),
            output_filename)
    print "Done."

    # QA clinic
    if QA:
        x = 32
        y = 32
        print "Starting QA clinic (free entrance to the masses)..."
        STC_QA(fmri_data, corrected_fmri_data, TR, x, y,
               compare_with=compare_with)


def demo_sinusoid(n_slices=10,
                  n_voxels_per_slice=1,
                  white_noise_std=1e-2,
                  artefact_std=4,
                  introduce_artefact_in_these_volumes="middle",
                  L=10,
                  ):
    """STC for time phase-shifted sinusoidal mixture in the presence of
    white-noise and volume-specific artefacts. This is supposed to be a
    BOLD time-course from a single voxel.

    Parameters
    ----------
    n_slices: int (optional, default 10)
        number of slices per 3D volume of the acquisition
    n_voxels_per_slice: int (optional, default 1)
        the number of voxels per slice in the simulated brain
        (setting this to 1 is sufficient for most QA since STC works
        slice-wise, and not voxel-wise)
    white_noise_std: float (optional, default 1e-2)
        amplitude of white noise to add to phase-shifted sample (spatial
        corruption)
    artefact_std: float (optional, default 4)
        amplitude of artefact
    introduce_artefact_in_these_volumesS: string, integer, or list of integers
    (optional, "middle")
        TR/volume index or indices to corrupt with an artefact (a spontaneous
        stray spike, probably due to instability of scanner B-field) of
        amplitude artefact_std
    L: int (optional, default None)
       Hanning Window width, passed to STC.fit(..) method

    """

    print "\r\n\t\t ---demo_sinusoid---"

    slice_indices = np.arange(n_slices, dtype=int)

    timescale = .01
    sine_freq = [.5, .8, .11, .7]  # number of complete cycles per unit time

    def my_sinusoid(t):
        """Creates mixture of sinusoids with different frequencies

        """

        res = t * 0

        for f in sine_freq:
            res += np.sin(2 * np.pi * t * f)

        return res

    time = np.arange(0, 24 + timescale, timescale)
    signal = my_sinusoid(time)

    # define timing vars
    freq = 10
    TR = freq * timescale

    # sample the time
    sampled_time = time[::freq]
    n_scans = len(sampled_time)

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_indices * slice_TR
    shifted_sampled_time = np.array([tau + sampled_time
                                     for tau in time_shift])

    # acquire the signal at the corrupt sampled time points
    acquired_signal = np.array([
            [my_sinusoid(shifted_sampled_time[j])
             for vox in xrange(n_voxels_per_slice)]
            for j in xrange(n_slices)])

    # add white noise
    acquired_signal += white_noise_std * np.random.randn(
        *acquired_signal.shape)

    # add artefacts to specific volumes/TRs
    if introduce_artefact_in_these_volumes is None:
        introduce_artefact_in_these_volumes = []
    if isinstance(introduce_artefact_in_these_volumes, int):
        introduce_artefact_in_these_volumes = [
            introduce_artefact_in_these_volumes]
    elif introduce_artefact_in_these_volumes == "middle":
        introduce_artefact_in_these_volumes = [n_scans / 2]
    else:
        assert hasattr(introduce_artefact_in_these_volumes, '__len__')
    introduce_artefact_in_these_volumes = np.array(
        introduce_artefact_in_these_volumes, dtype=int) % n_scans
    acquired_signal[:, :, introduce_artefact_in_these_volumes
                          ] += artefact_std * np.random.randn(
        n_slices,
        n_voxels_per_slice,
        len(introduce_artefact_in_these_volumes))

    # fit STC
    stc = STC()
    stc.fit(raw_data=acquired_signal, L=L)

    # apply STC
    print "Applying full-brain STC transform..."
    st_corrected_signal = stc.transform()
    print "Done."

    # QA clinic
    print "Starting QA clinic (free entrance to the masses)..."
    for slice_index in xrange(n_slices):
        for vox in xrange(n_voxels_per_slice):
            plot_slicetiming_results(
                TR,
                acquired_signal[slice_index][0],
                st_corrected_signal[slice_index][0],
                ground_truth_signal=signal,
                ground_truth_time=time,
                title=("Slice-Timing Correction of sampled sine mixeture "
                       "time-course from voxel %i of slice %i \nN.B:- "
                       "TR = %.2f, # slices = %i, # voxels per slice = %i, "
                       "white-noise std = %f, artefact std = %.2f, volumes "
                       "corrupt with artefact: %s, L: %s") % (
                    vox, slice_index, TR, n_slices, n_voxels_per_slice,
                    white_noise_std, artefact_std, ", ".join(
                        [str(i) for i in introduce_artefact_in_these_volumes]),
                    L)
                )

    # stc.show_slice_transform(4)

if __name__ == '__main__':
    # demo_sinusoid()
    # demo_HRF()
    demo_BOLD(dataset='spm-auditory',
              data_dir="/home/elvis/CODE/datasets/spm_auditory")
