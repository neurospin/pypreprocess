"""
:module: st
:synopsis: module for STC (Slice-Timing Correction) in fMRI data
:author: elvis[dot]dohmatob[at]inria[dot]fr

"""

import os
import nibabel
import scipy
import numpy as np
from nilearn.image.image import check_niimg
from .io_utils import is_niimg, save_vols, get_basenames


def get_slice_indices(n_slices, slice_order='ascending',
                      interleaved=False, return_final=False):
    """Function computes the (unique permutation on) slice indices, consistent
    with the specified slice order.

    Parameters
    ----------
    n_slices: int
        The number of slices there're altogether.

    slice_order: string ('ascending', 'descending'), or array of ints or
                 length n_slices slice order of acquisitions in a TR.
        'ascending': slices were acquired from bottommost to topmost
        'descending': slices were acquired from topmost to bottommost.
        If list of integers, 0-based (i.e Python!) indexing is assumed.

    interleaved: bool (optional, default False)
        Ff set, then slices were acquired in interleaved order, odd-numbered
        slices first, and then even-numbered slices.

    Returns
    -------
    slice_indices: 1D array of length n_slices
        Slice indices consistent with slice order (i.e, slice_indices[k]
        is the corrected index of slice k according to the slice order).

    Raises
    ------
    ValueError
    """

    if isinstance(slice_order, str):
        slice_indices = list(range(n_slices))
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
        if interleaved:
            raise ValueError(
                ("Since you have specified an explicit slice order, I don't "
                 "expecting you to set the 'interleaved' flag."))

        # here, I'm assuming an explicitly specified slice order as a
        # permutation on n symbols
        slice_order = np.array(slice_order, dtype='int')

        assert len(slice_order) == n_slices
        assert np.all((0 <= slice_order) & (
            slice_order < n_slices)), slice_order
        assert len(set(slice_order)) == n_slices, slice_order

        slice_indices = slice_order

    slice_indices = np.array(slice_indices)
    if return_final:
        slice_indices = np.array([np.nonzero(slice_indices == z)[0][0]
                                  for z in range(n_slices)])
    return slice_indices


class STC(object):
    """Correct differences in slice acquisition times.

    This correction assumes that the data are band-limited (i.e. there is
    no meaningful information present in the data at a frequency higher than
    that of the Nyquist). This assumption is supported by the study of Josephs
    et al (1997, NeuroImage) that obtained event-related data at an
    effective TR of 166 msecs. No physio-logical signal change was present
    at frequencies higher than their typical Nyquist (0.25 HZ).

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

    Attributes
    ----------
    kernel_: 2D array of shape (n_slices, n_scans)
        sinc kernel for phase shifting the different slices within each TR

    """

    def __init__(self, slice_order='ascending',
                 interleaved=False,
                 ref_slice=0,
                 verbose=1):

        # slice acquisition info
        self.slice_order = slice_order
        self.interleaved = interleaved
        self.ref_slice = ref_slice
        self.verbose = verbose

    def _log(self, msg):
        """Prints a message, according to the verbosity level.

        Parameters
        ----------
        msg: string
            the message to be printed

        """

        if self.verbose:
            print(msg)

    def __repr__(self):
        return str(self.__dict__)

    def _sanitize_raw_data(self, raw_data, fitting=False):
        """Checks that raw_data has shape that matches the fitted transform

        Parameters
        ----------
        raw_data: array-like
            raw data array being scrutinized

        fitting: bool, optional (default False)
            this flag indicates whether this method is being called from the
            fit(...) method (upon which ome special business will be handled)

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
                if raw_data.shape[2] != self.n_slices:
                    raise ValueError(
                        "raw_data has wrong number of slices: expecting %i,"
                        " got %i" % (self.n_slices, raw_data.shape[2]))

            # sanitize n_scans of raw data
            if hasattr(self, "_n_scans"):
                if raw_data.shape[3] != self.n_scans:
                    raise ValueError(
                        ("raw_data has wrong number of volumes: expecting %i, "
                         "got %i") % (self.n_scans, raw_data.shape[3]))

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
        self: fitted STC object

        Raises
        ------
        ValueError, in case parameters are insane

        """

        # set basic meta params
        if not raw_data is None:
            raw_data = self._sanitize_raw_data(raw_data, fitting=True,)
            self.n_slices = raw_data.shape[2]
            self.n_scans = raw_data.shape[-1]

            self.raw_data = raw_data
        else:
            if n_slices is None:
                raise ValueError(
                    "raw_data parameter not specified. You need to"
                    " specify a value for n_slices!")
            else:
                self.n_slices = n_slices
            if n_scans is None:
                raise ValueError(
                    "raw_data parameter not specified. You need to"
                    " specify a value for n_scans!")
            else:
                self.n_scans = n_scans

        # fix slice indices consistently with slice order
        self.slice_indices = get_slice_indices(
            self.n_slices, slice_order=self.slice_order, return_final=True,
            interleaved=self.interleaved)

        # fix ref slice index, to be consistent with the slice order
        self.ref_slice = self.slice_indices[self.ref_slice]

        # timing info (slice_TR is the time of acquisition of a single slice,
        # as a fractional multiple of the TR)
        if not timing is None:
            TR = (self.n_slices - 1) * timing[0] + timing[1]
            slice_TR = timing[0] / TR
            assert 0 <= slice_TR < 1
            self._log("Your TR is %s" % TR)
        else:
            # TR normalized to 1 (
            slice_TR = 1. / self.n_slices

        # least power of 2 not less than n_scans
        N = 2 ** int(np.floor(np.log2(self.n_scans)) + 1)

        # this will hold phase shifter of each slice k
        self.kernel_ = np.ndarray(
            (self.n_slices, N),
            dtype=np.complex,  # beware, default dtype is float!
            )

        # loop over slices (z axis)
        for z in range(self.n_slices):
            self._log(("STC: Estimating phase-shift transform for slice "
                       "%i/%i...") % (z + 1, self.n_slices))

            # compute time delta for shifting this slice w.r.t. the reference
            shift_amount = (
                self.slice_indices[z] - self.ref_slice) * slice_TR

            # phi represents a range of phases up to the Nyquist
            # frequency
            phi = np.ndarray(N)
            phi[0] = 0.
            for f in range(int(N / 2)):
                phi[f + 1] = -1. * shift_amount * 2 * np.pi * (f + 1) / N

            # check if signal length is odd or even -- impacts how phases
            # (phi) are reflected across Nyquist frequency
            offset = N % 2

            # mirror phi about the center
            phi[int(1 + N / 2 - offset):] = -phi[int(N / 2 + offset - 1):0:-1]

            # map phi to frequency domain: phi -> complex
            # point z = exp(i * phi) on unit circle
            self.kernel_[z] = scipy.cos(
                phi) + scipy.sqrt(-1) * scipy.sin(phi)

        self._log("Done.")

        # return fitted object
        return self

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
        self.output_data_: array of same shape as raw_data
            ST corrected data

        Raises
        ------
        Exception, if fit(...) has not yet been invoked

        """

        if self.kernel_ is None:
            raise Exception("fit(...) method not yet invoked!")

        # sanitize raw_data
        if raw_data is None:
            if hasattr(self, 'raw_data'):
                raw_data = self.raw_data
            else:
                raise RuntimeError(
                    'You need to specify raw_data that will be transformed.')

        raw_data = self._sanitize_raw_data(raw_data)

        n_rows, n_columns = raw_data.shape[:2]
        N = self.kernel_.shape[-1]

        # our workspace; organization is (extended) time x rows
        stack = np.ndarray((N, n_rows))

        # empty slate to hold corrected data
        self.output_data_ = 0 * raw_data

        # loop over slices (z axis)
        for z in range(self.n_slices):
            self._log(
                "STC: Correcting acquisition delay in slice %i/%i..." % (
                    z + 1, self.n_slices))

            # prepare phase-shifter for this slice
            shifter = np.array([self.kernel_[z], ] * n_rows).T

            # loop over columns of slice z (y axis)
            for y in range(n_columns):
                # extract column y of slice z of all 3D volumes
                stack[:self.n_scans, :] = raw_data[:, y, z, :].reshape(
                    (n_rows, self.n_scans)).T

                # fill-in continuous function to avoid edge effects (wrapping,
                # etc.): simply linspace the displacement between the start
                # and ending value of each BOLD response time-series
                for x in range(stack.shape[1]):
                    stack[self.n_scans:, x] = np.linspace(
                        stack[self.n_scans - 1, x], stack[0, x],
                        num=N - self.n_scans,).T

                # apply phase-shift to column y of slice z of all 3D volumes
                stack = np.real(np.fft.ifft(
                    np.fft.fft(stack, axis=0) * shifter, axis=0))

                # re-insert phase-shifted column y of slice z for all 3D
                # volumes
                self.output_data_[:, y, z, :] = stack[:self.n_scans,
                                                      :].T.reshape(
                    (n_rows, self.n_scans))

        self._log("Done.")

        # return output
        return self.output_data_

    def get_last_output_data(self):
        """Returns the output data computed by the last call to the transform
        method

        Raises
        ------
        Exception, if transform(...) has not yet been invoked

        """

        if self.output_data_ is None:
            raise Exception("transform(...) method not yet invoked!")

        return self.output_data_


class fMRISTC(STC):
    """
    Slice-Timing Correction for fMRI data.

    Attributes
    ----------
    kernel
    """

    def _sanitize_raw_data(self, raw_data, fitting=False):
        """
        Re-implementation of parent method to sanitize fMRI data.

        """

        if not hasattr(self, 'basenames_'):
            self.basenames_ = None

        if isinstance(raw_data, str):
            # str
            if isinstance(raw_data, str):
                self.basenames_ = os.path.basename(raw_data)
            img = nibabel.load(raw_data)
            raw_data, self.affine_ = img.get_data(), img.get_affine()
        elif is_niimg(raw_data):
            raw_data, self.affine_ = raw_data.get_data(), raw_data.get_affine()
        elif isinstance(raw_data, list) and (isinstance(
                raw_data[0], str) or is_niimg(raw_data[0])):
            # list of strings or niimgs
            if isinstance(raw_data[0], str):
                self.basenames_ = [os.path.basename(x) for x in raw_data]
            n_scans = len(raw_data)
            _first = check_niimg(raw_data[0])
            _raw_data = np.ndarray(list(_first.shape) + [n_scans])
            _raw_data[..., 0] = _first.get_data()
            self.affine_ = [_first.get_affine()]

            for t in range(1, n_scans):
                vol = check_niimg(raw_data[t])
                _raw_data[..., t] = vol.get_data()
                self.affine_.append(vol.get_affine())
            raw_data = _raw_data
        else:
            raw_data = np.array(raw_data)

        if raw_data.ndim == 5:
            assert raw_data.shape[-2] == 1, raw_data.shape
            raw_data = raw_data[..., 0, ...]

        # our business is over: deligate to super method
        return STC._sanitize_raw_data(self, raw_data, fitting=fitting)

    def get_raw_data(self):
        return self.raw_data

    def transform(self, raw_data=None, output_dir=None,
                  affine=None, prefix='a', basenames=None, ext=None):
        self.output_data_ = STC.transform(self, raw_data=raw_data)
        if not basenames is None:
            self.basenames_ = basenames
        if not affine is None:
            self.affine_ = affine
        if not output_dir is None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        if hasattr(self, 'affine_'):
            if isinstance(self.affine_, list):
                self.output_data_ = [nibabel.Nifti1Image(
                    self.output_data_[..., t], self.affine_[t])
                    for t in range(self.output_data_.shape[-1])]
                if output_dir is None:
                    self.output_data_ = nibabel.concat_images(
                        self.output_data_, check_affines=False)
            else:
                self.output_data_ = nibabel.Nifti1Image(self.output_data_,
                                                        self.affine_)
            if not output_dir is None:
                self.output_data_ = save_vols(
                    self.output_data_,
                    output_dir, prefix=prefix,
                    basenames=get_basenames(self.basenames_, ext=ext))
        return self.output_data_
