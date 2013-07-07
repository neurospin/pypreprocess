"""
:module: st
:synopsis: module for STC (Slice-Timing Correction) in fMRI data
:author: elvis[dot]dohmatob[at]inria[dot]fr

"""

import os
import nibabel
import scipy
import numpy as np
import matplotlib.pyplot as plt


def get_slice_indices(n_slices, slice_order='ascending',
                    interleaved=False,):
    """Function computes the slice indices, consistent with the
    specified slice order.

    Parameters
    ----------
    n_slices: int
        the number of slices there're altogether
    slice_order: string or array of ints or length n_slices
        slice order of acquisitions in a TR
        'ascending': slices were acquired from bottommost to topmost
        'descending': slices were acquired from topmost to bottommost
    interleaved: bool (optional, default False)
        if set, then slices were acquired in interleaved order, odd-numbered
        slices first, and then even-numbered slices

    Returns
    -------
    slice_indices: 1D array of length n_slices
        slice indices consistent with slice order (i.e, slice_indices[k]
        is the corrected index of slice k according to the slice order)

    Raises
    ------
    ValueError

    """

    if isinstance(slice_order, basestring):
        slice_indices = range(n_slices)
        if interleaved:
            # python indexing begins from 0 (MATLAB begins from 1)
            slice_indices = slice_indices[0::2] + slice_indices[1::2]
        if slice_order.lower() == 'ascending':
            pass
        elif slice_order.lower() == 'descending':
            slice_indices = np.flipud(slice_indices)
        else:
            raise ValueError("Unknown slice order '%s'!" % slice_order)
    else:
        # here, I'm assuming an explicitly specified slice order as a
        # permutation on n symbols
        slice_order = np.array(slice_order, dtype='int')

        assert len(slice_order) == n_slices
        assert np.all((0 <= slice_order) & (slice_order < n_slices))
        assert len(set(slice_order)) == n_slices

        slice_indices = slice_order

    slice_indices = np.array(slice_indices)
    slice_indices = np.array([np.nonzero(slice_indices == z)[0][0]
                              for z in xrange(n_slices)])

    return slice_indices


class STC(object):
    """Correct differences in slice acquisition times. This correction
    assumes that the data are band-limited (i.e. there is no meaningful
    information present in the data at a frequency higher than that of
    the Nyquist). This assumption is supported by the study of Josephs
    et al (1997, NeuroImage) that obtained event-related data at an
    effective TR of 166 msecs. No physio-logical signal change was present
    at frequencies higher than their typical Nyquist (0.25 HZ).

    """

    def __init__(self, slice_order='ascending',
                 interleaved=False,
                 ref_slice=0,
                 verbose=1):
        """Default constructor.

        Parameters
        ----------
        slice_order: string or array of ints or length n_slices
            slice order of acquisitions in a TR
            'ascending': slices were acquired from bottommost to topmost
            'descending': slices were acquired from topmost to bottommost
        interleaved: bool (optional, default False)
            if set, then slices were acquired in interleaved order,
            odd-numbered slices first, and then even-numbered slices
        ref_slice: int (optional, default 0)
            the slice number to be taken as the reference slice
        verbose: int (optional, default 1)
            verbosity level, set to 0 for no verbose

        """

        # slice acquisition info
        self._slice_order = slice_order
        self._interleaved = interleaved
        self._ref_slice = ref_slice

        self._verbose = verbose

        # holds phase shifters for the slices, one row per slice
        self._transform = None

        # this always holds the last output data produced by transform(..)
        # method
        self._output_data = None

    def _log(self, msg):
        """Prints a message, according to the verbosity level.

        Parameters
        ----------
        msg: string
            the message to be printed

        """

        if self._verbose:
            print(msg)

    def _sanitize_raw_data(self, raw_data, fitting=False):
        """Checks that raw_data has shape that matches the fitted transform

        Parameters
        ----------
        raw_data: array-like
            raw data array being scrutinized

        Returns
        -------
        raw_data: array
            sanitized raw_data

        Raises
        ------
        valueError if raw_data is badly shaped

        XXX TODO: add support for nifti images, or filenames

        """

        raw_data = np.array(raw_data)

        if len(raw_data.shape) != 4:
            raise ValueError(
                "raw_data must be 4D array, got %iD!" % len(raw_data.shape))

        # sanitize n_slices of raw_data
        if not fitting:
            if hasattr(self, "_n_slices"):
                if raw_data.shape[2] != self._n_slices:
                    raise ValueError(
                        "raw_data has wrong number of slices: expecting %i,"
                        " got %i" % (self._n_slices, raw_data.shape[2]))

            # sanitize n_scans of raw data
            if hasattr(self, "_n_scans"):
                if raw_data.shape[3] != self._n_scans:
                    raise ValueError(
                        ("raw_data has wrong number of volumes: expecting %i, "
                         "got %i") % (self._n_scans, raw_data.shape[3]))

        # return sanitized raw_dat
        return raw_data

    def fit(self, raw_data=None, n_slices=None, n_scans=None,
            timing=None,
            ):
        """Fits an STC transform that can be later used (using the
        transform(..) method) to re-slice compatible data.

        Each row of the fitter transform is precisely the filter by
        which the signal will be convolved to introduce the phase
        shift in the corresponding slice. It is constructed explicitly
        in the Fourier domain. In the time domain, it can be described
        via the Whittaker-Shannon formula (sinc interpolation).

        Parameters
        ----------
        raw_data: 4D array of shape (n_rows, n_colomns, n_slices,
        n_scans) (optional, default None)
            raw data to fit the transform on. If this is specified, then
            n_slices and n_scans parameters should not be specified.
        n_slices: int (optional, default None)
            number of slices in each 3D volume. If the raw_data parameter
            is specified then this parameter should not be specified
        n_scans: int (optional, default None)
            number of 3D volumes. If the raw_data parameter
            is specified then this parameter should not be specified
        timing: list or tuple of length 2 (optional, default None)
            additional information for sequence timing
            timing[0] = time between slices
            timing[1] = time between last slices and next volume

        Returns
        -------
        self._tranform: 2D array of shape (n_slices, least positive integer
        not less than self._n_scans)
            fft transform (phase shifts mapped into frequency domain). Each row
            is the filter by which the signal will be convolved to introduce
            the phase shift in the corresponding slice.

        Raises
        ------
        ValueError, in case parameters are insane

        """

        # set basic meta params
        if not raw_data is None:
            raw_data = self._sanitize_raw_data(raw_data, fitting=True,)
            self._n_slices = raw_data.shape[2]
            self._n_scans = raw_data.shape[-1]

            self._raw_data = raw_data
        else:
            if n_slices is None:
                raise ValueError(
                    "raw_data parameter not specified. You need to"
                    " specify a value for n_slices!")
            else:
                self._n_slices = n_slices
            if n_scans is None:
                raise ValueError(
                    "raw_data parameter not specified. You need to"
                    " specify a value for n_scans!")
            else:
                self._n_scans = n_scans

        # fix slice indices consistently with slice order
        self._slice_indices = get_slice_indices(self._n_slices,
                                                slice_order=self._slice_order,
                                                interleaved=self._interleaved,
                                                )

        # fix ref slice index, to be consistent with the slice order
        self._ref_slice = self._slice_indices[self._ref_slice]

        # timing info (slice_TR is the time of acquisition of a single slice,
        # as a fractional multiple of the TR)
        if not timing is None:
            TR = (self._n_slices - 1) * timing[0] + timing[1]
            slice_TR = timing[0] / TR
            assert 0 <= slice_TR < 1
            self._log("Your TR is %s" % TR)
        else:
            # TR normalized to 1 (
            slice_TR = 1. / self._n_slices

        # least power of 2 not less than n_scans
        N = 2 ** int(np.floor(np.log2(self._n_scans)) + 1)

        # this will hold phase shifter of each slice k
        self._transform = np.ndarray(
            (self._n_slices, N),
            dtype=np.complex,  # beware, default dtype is float!
            )

        # loop over slices (z axis)
        for z in xrange(self._n_slices):
            self._log(("STC: Estimating phase-shift transform for slice "
                       "%i/%i...") % (z + 1, self._n_slices))

            # compute time delta for shifting this slice w.r.t. the reference
            shift_amount = (
                self._slice_indices[z] - self._ref_slice) * slice_TR

            # phi represents a range of phases up to the Nyquist
            # frequency
            phi = np.ndarray(N)
            phi[0] = 0.
            for f in xrange(N / 2):
                phi[f + 1] = -1. * shift_amount * 2 * np.pi * (f + 1) / N

            # check if signal length is odd or even -- impacts how phases
            # (phi) are reflected across Nyquist frequency
            offset = N % 2

            # mirror phi about the center
            phi[1 + N / 2 - offset:] = -phi[N / 2 + offset - 1:0:-1]

            # map phi to frequency domain: phi -> complex
            # point z = exp(i * phi) on unit circle
            self._transform[z] = scipy.cos(
                phi) + scipy.sqrt(-1) * scipy.sin(phi)

        self._log("Done.")

        # return computed transform
        return self._transform

    def transform(self, raw_data=None):
        """
        Applies STC transform to raw data, thereby correcting for time-delay
        in acquisition.

        Parameters
        ----------
        raw_data: 4D array of shape (n_rows, n_columns, n_slices, n_scans),
        optional (default None)
            the data to be ST corrected. raw_data is Not modified in memory;
            another array is returned. If not specified, then the fitted
            data if used in place

        Returns
        -------
        self._output_data: array of same shape as raw_data
            ST corrected data

        Raises
        ------
        Exception, if fit(...) has not yet been invoked

        """

        if self._transform is None:
            raise Exception("fit(...) method not yet invoked!")

        # sanitize raw_data
        if raw_data is None:
            if hasattr(self, '_raw_data'):
                raw_data = self._raw_data
            else:
                raise RuntimeError(
                    'You need to specify raw_data that will be transformed.')

        raw_data = self._sanitize_raw_data(raw_data)

        n_rows, n_columns = raw_data.shape[:2]
        N = self._transform.shape[-1]

        # our workspace; organization is (extended) time x rows
        stack = np.ndarray((N, n_rows))

        # empty slate to hold corrected data
        self._output_data = 0 * raw_data

        # loop over slices (z axis)
        for z in xrange(self._n_slices):
            self._log(
                "STC: Correcting acquisition delay in slice %i/%i..." % (
                    z + 1, self._n_slices))

            # prepare phase-shifter for this slice
            shifter = np.array([self._transform[z], ] * n_rows).T

            # loop over columns of slice z (y axis)
            for y in xrange(n_columns):
                # extract column y of slice z of all 3D volumes
                stack[:self._n_scans, :] = raw_data[:, y, z, :].reshape(
                    (n_rows, self._n_scans)).T

                # fill-in continuous function to avoid edge effects (wrapping,
                # etc.): simply linspace the displacement between the start
                # and ending value of each BOLD response time-series
                for x in xrange(stack.shape[1]):
                    stack[self._n_scans:, x] = np.linspace(
                        stack[self._n_scans - 1, x], stack[0, x],
                        num=N - self._n_scans,).T

                # apply phase-shift to column y of slice z of all 3D volumes
                stack = np.real(np.fft.ifft(
                        np.fft.fft(stack, axis=0) * shifter, axis=0))

                # re-insert phase-shifted column y of slice z for all 3D
                # volumes
                self._output_data[:, y, z, :] = stack[:self._n_scans,
                                                       :].T.reshape(
                    (n_rows, self._n_scans))

        self._log("Done.")

        # return output
        return self._output_data

    def get_last_output_data(self):
        """Returns the output data computed by the last call to the transform
        method

        Raises
        ------
        Exception, if transform(...) has not yet been invoked

        """

        if self._output_data is None:
            raise Exception("transform(...) method not yet invoked!")

        return self._output_data


class fMRISTC(STC):
    def _sanitize_raw_data(self, raw_data, **kwargs):
        """
        Re-implementation of parent method to sanitize fMRI data.

        """

        if isinstance(raw_data, np.ndarray) or isinstance(raw_data, list) \
                or isinstance(raw_data, basestring):
            raw_data = self._load_fmri_data(raw_data)

        return STC._sanitize_raw_data(self, raw_data, **kwargs)

    def _load_fmri_data(self, fmri_files, is_3D=False):
        """
        Helper function to load fmri data from filename /
        ndarray or list of such.

        Parameters
        ----------
        fmri_files: `np.ndarray` or string of list of strings, or (recursive)
        of list of such
            the data to be loaded.
            if string, it should be the filename of a single 3D vol or 4D
            fmri film.
        is_3D: boolean
            flag specifying whether loaded data is in fact 3D. This is useful
            for loading volumes with shapes like (25, 34, 56, 1), where the
            last dimension can be ignored altogether

        Returns
        -------
        data: `np.ndarray`
            the loaded data

        """

        if isinstance(fmri_files, np.ndarray):
            data = fmri_files
        elif isinstance(fmri_files, basestring):
            data = nibabel.load(fmri_files).get_data()
        else:
            # assuming list of (perhaps list of ...) filenames (strings)
            n_scans = len(fmri_files)
            _first = self._load_fmri_data(fmri_files[0], is_3D=True)
            data = np.ndarray(tuple(list(_first.shape
                                         ) + [n_scans]))
            data[..., 0] = _first
            for scan in xrange(1, n_scans):
                data[..., scan] = self._load_fmri_data(fmri_files[scan],
                                                       is_3D=True)

        if is_3D:
            if data.ndim == 4:
                data = data[..., 0]

        return data

    def get_raw_data(self):
        return self._raw_data

    # def _save_stc_output(self, output_dir,
    #                      input_filenames,
    #                      prefix='a'):

    #     self._log("Saving STC output to %s..." % output_dir)

    #     # sanitize output_diir
    #     ref_filename = input_filenames if isinstance(
    #         input_filenames, basestring) else input_filenames[0]
    #     ref_file_basename = os.path.basename(ref_filename)
    #     if output_dir is None:
    #         output_dir = output_dir
    #     if output_dir is None:
    #         output_dir = os.path.dirname(ref_filename)
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     # save corrected image files to disk
    #     if isinstance(input_filenames, basestring):
    #         affine = nibabel.load(input_filenames).get_affine()

    #         for t in xrange(self._n_scans):
    #             output_filename = os.path.join(output_dir,
    #                                            "%s%i%s" % (
    #                     prefix, t, ref_file_basename))

    #             nibabel.save(nibabel.Nifti1Image(
    #                     self.get_last_output_data[..., t],
    #                     affine),
    #                          output_filename)

    #         output_filenames = output_filename
    #     else:
    #         output_filenames = []
    #         for filename, t in zip(input_filenames, xrange(self._n_scans)):
    #             affine = nibabel.load(filename).get_affine()
    #             output_filename = os.path.join(output_dir,
    #                                            "%s%s" % (
    #                     prefix,
    #                     os.path.basename(filename)))

    #             nibabel.save(nibabel.Nifti1Image(
    #                     self.get_last_output_data[..., t],
    #                     affine),
    #                          output_filename)

    #             output_filenames.append(output_filename)

    #     return output_filenames


def plot_slicetiming_results(acquired_sample,
                             st_corrected_sample,
                             TR=1.,
                             ground_truth_signal=None,
                             ground_truth_time=None,
                             x=None,
                             y=None,
                             compare_with=None,
                             suptitle_prefix="",
                             output_dir=None,
                             ):
    """Function to generate QA plots post-STC business, for a single voxel

    Parameters
    ----------
    acquired_sample: 1D array
        the input sample signal to the STC
    st_corrected_sample: 1D array same shape as
    acquired_sample
        the output corrected signal from the STC
    TR: float
        Repeation Time exploited by the STC algorithm
    ground_truth_signal: 1D array (optional, default None), same length as
    acquired_signal
        ground truth signal
    ground_truth_time: array (optional, default None), same length as
    ground_truth_time
        ground truth time w.r.t. which the ground truth signal was collected
    x: int (optional, default None)
        x coordinate of test voxel used for QA
    y: int (optional, default None)
        y coordinate of test voxel used for QA
    compare_with: 1D array of same shape as st_corrected_array (optional,
    default None)
        output from another STC implementation, so we can compare ours
        that implementation
    suptitle_prefix: string (optional, default "")
        prefix to append to suptitles
    output_dir: string, optional (default None)
        dirname where generated plots will be saved

    Returns
    -------
    None

    """

    # sanitize arrays
    acquired_sample = np.array(acquired_sample)
    st_corrected_sample = np.array(st_corrected_sample)

    n_rows, n_columns, n_slices, n_scans = acquired_sample.shape

    if not compare_with is None:
        compare_with = np.array(compare_with)
        assert compare_with.shape == acquired_sample.shape

    # centralize x and y if None
    x = n_rows / 2 if x is None else x
    y = n_columns / 2 if y is None else y

    # sanitize x and y
    x = x % n_rows
    y = y % n_columns

    # number of rows in plot
    n_rows_plot = 2

    if not ground_truth_signal is None and not ground_truth_time is None:
        n_rows_plot += 1
        N = len(ground_truth_signal)
        sampling_freq = (N - 1) / (n_scans - 1)  # XXX formula correct ??

        # acquire signal at same time points as corrected sample
        sampled_ground_truth_signal = ground_truth_signal[
            ::sampling_freq]

    print ("Starting QA engines for %i voxels in the line x = %i, y = %i"
           " (close figure to see the next one)..." % (n_slices, x, y))

    acquisition_time = np.linspace(0, (n_scans - 1) * TR, n_scans)
    for z in xrange(n_slices):
        # setup for plotting
        plt.figure()
        plt.suptitle('%s: QA for voxel %s' % (suptitle_prefix, str((x, y, z))))

        ax1 = plt.subplot2grid((n_rows_plot, 1),
                               (0, 0))

        # plot acquired sample
        ax1.plot(acquisition_time, acquired_sample[x][y][z],
                 'r--o')
        ax1.hold('on')

        # plot ST corrected sample
        ax1.plot(acquisition_time, st_corrected_sample[x][y][z],
                 's-')
        ax1.hold('on')

        # plot groud-truth (if provided)
        if not ground_truth_signal is None and not ground_truth_time is None:
            ax1.plot(ground_truth_time, ground_truth_signal)
            plt.hold('on')

            ax3 = plt.subplot2grid((n_rows_plot, 1),
                                   (2, 0))

            # compute absolute error and plot an error
            abs_error = np.abs(
                sampled_ground_truth_signal - st_corrected_sample[x][y][z])
            ax3.plot(acquisition_time, abs_error)
            ax3.hold("on")

            # compute and plot absolute error for other method
            if not compare_with is None:
                compare_with_abs_error = np.abs(
                    sampled_ground_truth_signal - compare_with[x][y][z])
                ax3.plot(acquisition_time, compare_with_abs_error)
                ax3.hold("on")

        if not compare_with is None:
            ax1.plot(acquisition_time, compare_with[x][y][z],
                     's-')
            ax1.hold('on')

        # plot ffts
        # XXX the zeroth time point has been removed in the plots below
        # to enable a better appretiation of the y axis
        ax2 = plt.subplot2grid((n_rows_plot, 1),
                               (1, 0))

        ax2.plot(acquisition_time[1:],
                 np.abs(np.fft.fft(acquired_sample[x][y][z])[1:]))

        ax2.plot(acquisition_time[1:],
                 np.abs(np.fft.fft(st_corrected_sample[x][y][z])[1:]))

        if not compare_with is None:
            ax2.plot(acquisition_time[1:],
                     np.abs(np.fft.fft(compare_with[x][y][z])[1:]))

        # misc
        plt.xlabel("time (s)")

        method1 = "ST corrected sample"
        if not compare_with is None:
            method1 = "STC method 1"

        ax1.legend(("Acquired sample",
                    method1,
                    "STC method 2",
                    "Ground-truth signal",))
        ax1.set_ylabel("BOLD")

        # ax2.set_title("Absolute value of FFT")
        ax2.legend(("Acquired sample",
                    method1,
                    "STC method 2"))
        ax2.set_ylabel("|fft|")

        if n_rows_plot > 2:
            # ax3.set_title(
            #     "Absolute Error (between ground-truth and correctd sample")
            ax3.legend((method1,
                        "STC method 2",))
            ax3.set_ylabel("absolute error")

        if not output_dir is None:
            output_filename = os.path.join(output_dir,
                                           "stc_results__slice_%i.png" % z)
            # dump image unto disk
            plt.savefig(output_filename, bbox_inches="tight", dpi=200)

        plt.show()

    print "Done."
