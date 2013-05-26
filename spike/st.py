"""
:module: st
:synopsis: module for STC (Slice-Timing Correction) in fMRI data
:author: elvis[dot]dohmatob[at]inria[dot]fr

"""

import os
import sys
import nibabel as ni
import scipy
import numpy as np
import unittest
import matplotlib.pyplot as plt


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


class STC(object):
    def fit(self, n_slices, n_scans, slice_order='ascending',
            interleaved=False,
            ref_slice=0,
            ):
        """Fits an STC transform that can be later used (using the
        transform(..) method) to re-slice compatible data

        Parameters
        ----------
        n_slices: int
            number of slices in each 3D volume
        n_scans: int
            number of 3D volumes
        slice_order: string or array of ints or length `n_slices`
            slice order of acquisitions in a TR
            'ascending': slices were acquired from bottommost to topmost
            'descending': slices were acquired from topmost to bottommost
        interleaved: bool (optional, default False)
            if set, then slices were acquired in interleaved order,
            odd-numbered slices first, and then even-numbered slices
        ref_slice: int (optional, default 0)
            the slice number to be taken as the reference slice

        """

        # set basic meta params
        self._n_slices = n_slices
        self._n_scans = n_scans
        self._slice_order = slice_order
        self._interleaved = interleaved
        self._ref_slice = ref_slice

        # fix slice indices consistently with slice order
        self._slice_indices = get_slice_indices(self._n_slices,
                                                slice_order=self._slice_order,
                                                interleaved=self._interleaved,
                                                )

        # fix ref slice index, to be consistent with the slice order
        # (acquisition type)
        self._ref_slice = self._slice_indices[self._ref_slice]

        # least power of 2 not less than n_scans
        N = 2 ** int(np.floor(np.log2(self._n_scans)) + 1)

        # time of acquisition of a single slice
        factor = 1. / self._n_slices

        # this will hold time shifter of each slice
        self._transform = np.zeros(
            (self._n_slices, N),
            dtype=np.complex,  # beware, default dtype if float!
            )

        for k in xrange(self._n_slices):
            print "STC: Estimating time-shift transform for slice %i/%i..." % (
                k + 1,
                self._n_slices)

            # set up time acquired within order
            shiftamount = (self._slice_indices[k] - self._ref_slice) * factor

            # phi represents a range of phases up to the Nyquist
            # frequency
            phi = np.zeros(N)
            for f in xrange(N / 2):
                # In the spm_slice_timing.m source code, the line below
                # reads: "phi(f+1) = -1*shiftamount*2*pi/(len/f)".
                # However, we can't do this in python since the expression
                # (len / f) first evaluated to machine precision with
                # floating-point error, before the "(f + 1) /" division is
                # performed, leading to a final result which is severely
                # prune to error!
                phi[f + 1] = -1. * shiftamount * 2 * np.pi * (f + 1) / N

            # check if signal length is odd or even -- impacts how phases
            # (phi) are reflected across Nyquist frequency
            offset = N % 2

            # mirror phi about the center
            phi[1 + N / 2 - offset:] = -np.flipud(phi[1:N / 2 + offset])

            # map phi to frequency domain: phi -> corresponding complex
            # point z on unit circle
            self._transform[k] = scipy.cos(
                phi) + scipy.sqrt(-1) * scipy.sin(phi)

        print "Done."

        # return computed transform
        return self._transform

    def _sanitize_raw_data(self, raw_data):
        """Checks that raw_data has shape that matches the fitted transform

        Parameters
        ----------
        raw_data: array
            raw data array being scrutinized

        Raises
        ------
        Exception if raw_data is badly shaped

        """

        if len(raw_data.shape) != 4:
            raise Exception("raw_data must be 4D array")

        if raw_data.shape[2] != self._n_slices:
            raise Exception(
                "raw_data has wrong number of slices: expecting %i, got %i" % (
                    self._n_slices,
                    raw_data.shape[2]))

        if raw_data.shape[3] != self._n_scans:
            raise Exception(
                ("raw_data has wrong number of volumes: expecting %i, "
                 "got %i") % (self._n_scans, raw_data.shape[3]))

    def transform(self, raw_data):
        """Applies an STC transform to raw data

        Parameters
        ----------
        raw_data: 4D array of shape (n_rows, n_columns, n_slices, n_scans)
            the data to be ST corrected

        Returns
        -------
        output_data: array of same shape as raw_data
            ST corrected data

        Notes
        -----
        Yes, there are quiet a bunch of for loops that may seem dull to you,
        and you might be legally tempted to vectorize 'em. Beware of
        voodoo vectorizations though, for those loops clumsy looking for
        ensure an ecological memory foot-print.

        """

        # sanitize raw_data
        self._sanitize_raw_data(raw_data)

        n_rows, n_columns = raw_data.shape[:2]
        N = self._transform.shape[-1]

        # our workspace; organization is (extended) time x rows
        stack = np.zeros((N, n_rows))

        # empty slate to hold corrected data
        self._output_data = 0 * raw_data

        # loop over slices (z axis)
        for z in xrange(self._n_slices):
            print "STC: Correcting acquisition delay in slice %i/%i..." % (
                z + 1, self._n_slices)

            # get time-shifter for slice z
            shifter = self._transform[z].copy()  # copy to avoid corruption

            # replicate shifter as many times as there are rows,
            # and then conjugate the result
            shifter = np.array([shifter, ] * n_rows).T

            # loop over columns of slice z (y axis)
            for y in xrange(n_columns):
                # extract column i of slice k of all 3D volumes
                stack[:self._n_scans, :] = raw_data[:, y, z, :].reshape(
                    (n_rows, self._n_scans)).T

                # fill-in continuous function to avoid edge effects
                # the technique is to simply linspace the displacement between
                # the start and ending value of the BOLD response
                for x in xrange(stack.shape[1]):
                    stack[self._n_scans:, x] = np.linspace(
                        stack[self._n_scans - 1, x], stack[0, x],
                        num=N - self._n_scans,).T

                # time-shift column y of slice z of all 3D volumes
                stack = np.real(np.fft.ifft(
                        np.fft.fft(stack, axis=0) * shifter, axis=0))

                # re-insert time-shifted column y of slice z for all 3D volumes
                self._output_data[:, y, z, :] = stack[:self._n_scans,
                                                       :].T.reshape(
                    (n_rows,
                     self._n_scans))

        print "Done."

        # return output
        return self._output_data

    def get_last_output_data(self):
        return self._output_data

    def get_output_data(self):
        return self.get_last_output_data()


def demo_sinusoid_mixture(n_slices=4, n_rows=1, n_columns=1,
                          introduce_artefact_in_these_volumes=None,
                          artefact_std=4.,
                          white_noise_std=1e-2,
                          ):

    n_voxels_per_slice = n_rows * n_columns

    print "\r\n\t\t ---demo_sinusoid---"

    slice_indices = np.arange(n_slices, dtype=int)

    timescale = .01
    sine_freq = [.5, .8, .11,
                  .7]  # number of complete cycles per unit time

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

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_indices * slice_TR
    shifted_sampled_time = np.array([tau + sampled_time
                                     for tau in time_shift])

    # acquire the signal at the corrupt sampled time points
    acquired_signal = np.array([
            [[my_sinusoid(shifted_sampled_time[j])
              for j in xrange(n_slices)]
             for y in xrange(n_columns)] for x in xrange(n_rows)]
                               )

    # add white noise
    acquired_signal += white_noise_std * np.random.randn(
        *acquired_signal.shape)

    n_scans = len(sampled_time)

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
    acquired_signal[:, :, :, introduce_artefact_in_these_volumes
                          ] += artefact_std * np.random.randn(
        n_rows,
         n_columns,
         len(introduce_artefact_in_these_volumes))

    # fit STC
    stc = STC()
    stc.fit(n_slices, n_scans)

    # apply STC
    st_corrected_signal = stc.transform(acquired_signal)

    for slice_index in xrange(n_slices):
        for x in xrange(n_rows):
            for y in xrange(n_columns):
                title = (
                    "Slice-Timing Correction of sampled sine mixeture "
                    "time-course from voxel %s of slice %i \nN.B:- "
                    "TR = %.2f, # slices = %i, # voxels per slice = %i, "
                    "white-noise std = %f, artefact std = %.2f") % (
                    str((x, y)),
                    slice_index, TR, n_slices,
                    n_voxels_per_slice,
                    white_noise_std, artefact_std,
                    )

                plt.plot(time, signal)
                plt.hold('on')
                plt.plot(sampled_time, acquired_signal[x][y][slice_index],
                         'r--o')
                plt.hold('on')
                plt.plot(sampled_time,
                         st_corrected_signal[x][y][slice_index],
                         's-')
                plt.hold('on')

                # misc
                plt.title(title)
                plt.legend(("Ground-truth signal", "Acquired sample",
                            "ST corrected sample"))
                plt.xlabel("time (s)")
                plt.ylabel("BOLD")

                plt.show()


def demo_real_BOLD(dataset='localizer',
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
        data_path = os.path.join(
            os.environ["HOME"],
            ".nipy/tests/data/s12069_swaloc1_corr.nii.gz")
        if not os.path.exists(data_path):
            raise Exception("You don't have nipy test data installed!")

        fmri_img = ni.load(data_path)
        fmri_data = fmri_img.get_data()

        TR = 2.4
    elif dataset == 'face-rep-spm5':
        # XXX nibabel says the affines of the 3Ds are different
        if not os.path.basename(os.environ["HOME"]) in ["elvis", "edohmato"]:
            raise Exception("Oops! This demo will sure fail on your PC!")
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

    print "\r\n\t\t ---demo_real_BOLD (%s)---" % dataset

    # fit STC
    n_scans = fmri_data.shape[-1]
    n_slices = fmri_data.shape[2]
    stc = STC()
    stc.fit(n_slices, n_scans,
            slice_order=slice_order,
            interleaved=interleaved,
            )

    # do full-brain ST coon
    corrected_fmri_data = stc.transform(fmri_data)

    # save output unto disk
    print "Saving ST corrected image to %s..." % output_filename
    ni.save(ni.Nifti1Image(corrected_fmri_data, fmri_img.get_affine()),
            output_filename)
    print "Done."

    # QA clinic
    if QA:
        sampled_time = np.linspace(0, (n_scans - 1) * TR, n_scans)
        for z in xrange(n_slices):
            ax1 = plt.subplot2grid((2, 1),
                                   (0, 0))
            # plot acquired sample
            ax1.plot(sampled_time, fmri_data[32][32][z],
                     'r--o')
            ax1.hold('on')

            # plot ST corrected sample
            ax1.plot(sampled_time, corrected_fmri_data[32][32][z],
                     's-')
            ax1.hold('on')

            if not compare_with is None:
                ax1.plot(sampled_time, compare_with[32][32][z],
                         's-')
                ax1.hold('on')

            # plot ffts
            ax2 = plt.subplot2grid((2, 1),
                                   (1, 0))

            ax2.plot(sampled_time[1:],
                     np.abs(np.fft.fft(fmri_data[32][32][z])[1:]))

            ax2.plot(sampled_time[1:],
                     np.abs(np.fft.fft(corrected_fmri_data[32][32][z])[1:]))

            if not compare_with is None:
                ax2.plot(sampled_time[1:],
                         np.abs(np.fft.fft(compare_with[32][32][z])[1:]))

            # misc
            ax1.set_title("Data")
            ax1.legend(("Acquired sample",
                        "STC method 1",
                        "STC method 2"))
            ax1.set_ylabel("BOLD")
            ax2.set_title("Absolute value of FFT")
            ax2.legend(("Acquired sample",
                        "STC method 1",
                        "STC method 2"))
            ax2.set_ylabel("energy")
            plt.xlabel("time (s)")

            # show generated plots
            plt.show()


if __name__ == '__main__':
    demo_sinusoid_mixture()
    demo_real_BOLD(dataset="localizer")
