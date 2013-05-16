"""
:Module: slice_timing.py
:Synopsis: Module for STC
:Author: elvis[dot]dohmatob[at]inria[dot]fr

"""

import numpy as np
import math
import pylab as pl
import matplotlib.pyplot as plt
import scipy
import scipy.signal
from nipy.algorithms.registration._registration\
    import _cspline_sample1d, _cspline_transform


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


def lanczos_window(t, a=2.):
    return 0. if np.abs(t) > a else scipy.sinc(t / a)


def do_slicetiming(acquired_signal, TR, n_slices, slice_index,
                   user_time=None,
                   slice_order='ascending',
                   interleaved=False,
                   interpolator='alexisroche_cspline'  # 'whittaker_shannon',
                   ):
    """Function does sinc temporal interpolation with Hanning window of
    prescribed width.

    Parameters
    ----------
    acquired_signal: array-like
        1D array of BOLD values (time-course from single voxel) or 2D array
        of BOLD values (time-courses from multiple voxels in thesame
        slice, one time-course per row) to be temporally interpolated

    TR: float
        Repeation Time at which acquired_signal was sampled from the
        unknown signal

    n_slices: int
        number of slices per TR

    slice_index: int
        index of this slice in the bail of slices, according to the respective
        acquisition order

    slice_order: string or array of ints or length `n_slices`
        slice order of acquisitions in a TR
        'ascending': slices were acquired from bottommost to topmost
        'descending': slices were acquired from topmost to bottommost

    interleaved: bool (optional, default False)
        if set, then slices were acquired in interleaved order, odd-numbered
        slices first, and then even-numbered slices

    interpolator: string (optional, default cspline)
        interpolator to use for STC. Pssible values are
        'whittaker_shannon':
            use my implementation of the Whittaker-Shannon formula
        'alexisroche_cspline':
            use Alexis Roche's nipy cbspline implementation

        'scipy_cspline': use scipy's signal.cspline_sample1d interpolator

    Returns
    -------
    array: ST corrected signal, same shape as input `acquired_signal`

    """

    assert 0 <= slice_index < n_slices

    # dim sanity checks
    if not len(acquired_signal.shape) == 1:
        assert len(acquired_signal.shape) == 2
    n_scans = acquired_signal.shape[-1]

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

    print "[+] Reslicing slice %i/%i..." % (slice_index + 1, n_slices)

    # (average) acquisition time for a slice (as a multiple of TR)
    slice_TR_factor = 1. / n_slices

    # acquisition time for voxels in this slice, as assumed by GLM, etc.
    TA = n_scans - 1
    assumed_acquisition_time = np.linspace(0, TA, n_scans)

    # XXX user should specify this instants (well, the below should
    # be default)
    if user_time is None:
        user_time = assumed_acquisition_time - slice_index * slice_TR_factor

    # correct the acquisition sample by shifting it in time
    # slice_index * slice_TR time units to the left to match the assumed
    # acquisition times
    if interpolator == 'whittaker_shannon':
        # XXX dig into applying "mirror-symmetric boundary conditions"

        # # set up shifting variables
        # shiftamount = slice_index * slice_TR
        # _len = 2 * n_scans
        # phi = np.zeros(_len)

        # # check parity of n_scans -- impacts the way phi if mirrowed
        # offset = n_scans % 2

        # # phi represents a range of phases up to the Nyquist frequency
        # for f in xrange(_len / 2):
        #     phi[f + 1] = -1 * shiftamount * 1. / (_len / (f + 1))

        # # mirrow phi about the center
        # phi[_len / 2 - offset:_len] = -np.flipud(
        #     phi[1:_len / 2 + 1 + offset])

        # pl.plot(phi)

        # # transform phi to frequency domain and take complex transpose
        # shifter = (np.cos(phi) + np.sin(phi) * scipy.sqrt(-1))

        # def reflect(x, a=-1):
        #     assert len(x.shape) == 1

        #     return np.hstack((a * np.flipud(x[1:]), x))

        # print shifter
        # stack = np.zeros(_len)
        # stack[:n_scans] = acquired_signal

        # stack = np.real(np.fft.ifft(np.fft.fft(
        #             stack, axis=0) * shifter, axis=0))

        # return stack[:n_scans]
        # pl.plot(phi); pl.show()

        # # auxiliary array of time shifts
        # time_deltas = np.array(
        #     [phi[:-1]
        #      for i in xrange(n_scans)])

        # acquired_signal = reflect(acquired_signal, 1)
        # print time_deltas.shape

        def reflected_points(x, y):
            assert len(x.shape) == len(y.shape) == 1
            assert len(x) == len(y)

            _x = np.hstack((-np.flipud(x[1:]), x))
            _y = np.hstack((np.flipud(y[1:]), y))

            return _x, _y

        def reflected(x, a=-1):
            _x = np.hstack((a * np.flipud(x[1:]), x))

            return _x

        if len(acquired_signal.shape) == 1:
            assumed_acquisition_time, acquired_signal = reflected_points(
                assumed_acquisition_time, acquired_signal)
        else:
            assumed_acquisition_time = reflected(assumed_acquisition_time)
            _tmp = np.zeros((acquired_signal.shape[0], 2 * n_scans - 1))
            for j in xrange(acquired_signal.shape[0]):
                _tmp[j, :] = reflected(acquired_signal[j, :], a=1)
            acquired_signal = _tmp

            # pl.plot(assumed_acquisition_time, acquired_signal[2, :])
            # pl.show()

        # XXX debug the commented code stub above!!! -- elvis
        time_deltas = np.array([
                user_time[j] - assumed_acquisition_time
                for j in xrange(len(user_time))])

        L = 10  # width of Hanning window

        # Do sinc interpolation with Hanning window (faster and theoretically
        # prevents artefacts to 'teleport' themselves accross TR's)
        # XX However, sinc interpolation is known to cause severe ringing due
        # discontinuous edges in the input data sample
        st_corrected_signal = np.dot(
             acquired_signal,
             (scipy.sinc(time_deltas)#  * np.vectorize(hanning_window)(
                    # time_deltas, L
                    # )
              ).T,
             )

        return st_corrected_signal

    elif interpolator == 'alexisroche_cspline':
        st_corrected_signal = 0 * acquired_signal

        if len(acquired_signal.shape) > 1:
            # XXX results are utterly wrong!
            # alexis's (back-end) cbspline code overrides the input buffer.
            # Worst still, it segfaults saying "/usr/lib/python2.7/...
            # dist-packages/scipy/integrate/vode.soAborted (core dumped)"
            for j in xrange(acquired_signal.shape[0]):
                tmp = st_corrected_signal[j, :] * 0
                st_corrected_signal[j, :] = _cspline_sample1d(
                    tmp,
                    _cspline_transform(acquired_signal[j, :]),
                    user_time,
                    )
        else:
            st_corrected_signal = _cspline_sample1d(
                st_corrected_signal,
                _cspline_transform(acquired_signal),
                user_time,
                )
    elif interpolator == 'scipy_cspline':
        # XXX this is dreadfully slow!!!
        # seems to lead to more activation being 'reported'
        if len(acquired_signal.shape) > 1:
            st_corrected_signal = np.array([
                    # scipy.interpolate.splev(
                    #     user_time,
                    #     scipy.interpolate.splrep(
                    #         assumed_acquisition_time,
                    #         acquired_signal[j, :],
                    #         )
                    #     )
                    scipy.signal.cspline1d_eval(
                        scipy.signal.cspline1d(acquired_signal[j, :]),
                        user_time,
                        dx=assumed_acquisition_time[
                            1] - assumed_acquisition_time[0],
                        x0=assumed_acquisition_time[0],
                        )
                    for j in np.arange(acquired_signal.shape[0])])
        else:
            # st_corrected_signal = scipy.interpolate.splev(
            #     user_time,
            #     scipy.interpolate.splrep(
            #         assumed_acquisition_time,
            #         acquired_signal,
            #         )
            #     )
            st_corrected_signal = scipy.signal.cspline1d_eval(
                scipy.signal.cspline1d(acquired_signal),
                user_time,
                dx=assumed_acquisition_time[
                    1] - assumed_acquisition_time[0],
                x0=assumed_acquisition_time[0],
                )
    else:
        raise RuntimeError("Unimplemented interpolator: %s" % interpolator)

    # sanitize output shape
    # st_corrected_signal = st_corrected_signal.reshape(acquired_signal.shape)

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

    assumed_acquisition_time = np.linspace(0, (n_scans - 1) * TR, n_scans)

    N = None
    if ground_truth_signal is None:
        ground_truth_signal = st_corrected_signal
    else:
        N = len(ground_truth_signal)

    if ground_truth_time is None:
        assert len(ground_truth_signal) == len(assumed_acquisition_time)
        ground_truth_time = assumed_acquisition_time

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
    ax1.plot(assumed_acquisition_time, acquired_signal, 'r--o')
    ax1.hold('on')

    # plot ST corrected signal
    ax1.plot(assumed_acquisition_time, st_corrected_signal, 'gs')
    ax1.hold('on')

    # misc
    if title:
        ax1.set_title(title)
    ax1.legend(('Ground-Truth signal',
               'Input sample',
               'Output ST corrected sample',
               ),
              loc='best')

    # plot squared error
    if not N is None:
        sampling_freq = (N - 1) / (n_scans - 1)  # XXX formula correct ??

        # acquire signal at same time points as corrected sample
        sampled_ground_truth_signal = ground_truth_signal[
            ::sampling_freq]

        ax2 = plt.subplot2grid((3, 1), (2, 0),
                               rowspan=1)
        ax2.set_title(
            "Absolute Error (between ground-truth and corrected sample)")
        ax2.plot(assumed_acquisition_time,
                np.abs(sampled_ground_truth_signal - st_corrected_signal))

    plt.xlabel('time')

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

    # acquire the signal at the corrupt sampled time pointss
    acquired_signal = compute_hrf(shifted_sampled_time,
                                  )

    # add white noise
    acquired_signal += white_noise_std * np.random.randn(
        len(acquired_signal))

    # do STC
    st_corrected_signal = do_slicetiming(
        acquired_signal, TR, n_slices=n_slices,
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
        print fmri_files
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
                TR, n_slices=n_slices,
                slice_index=z,
                slice_order=slice_order,
                interpolator='whittaker_shannon',
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

    print "\r\n[+] Wrote ST corrected image to %s\r\n" % output_filename

    # # QA clinic
    # for z in slice_index:
    #     x, y = 32, 32
    #     plot_slicetiming_results(
    #         TR,
    #         fmri_data[x, y, z, :],
    #         corrected_fmri_data[x, y, z, :],
    #         title=("Slice-Timing Correction BOLD time-course from a single"
    #                "voxel \nN.B:- TR = %.2f, # slices = %i, x = %i, y = %i,"
    #                " slice index (z) = %i") % (TR, n_slices, x, y, z)
    #         )


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
        acquired_signal, TR, n_slices=n_slices,
        slice_index=slice_index,
        interpolator='whittaker_shannon',
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
    demo_BOLD(dataset='face_rep_SPM5')
    demo_sinusoid(n_slices=4, slice_index=-1)
    demo_HRF(slice_index=-1)
