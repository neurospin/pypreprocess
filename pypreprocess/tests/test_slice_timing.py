import os
import inspect
import pytest
import numpy as np
import nibabel
from ..slice_timing import STC, fMRISTC, get_slice_indices
from ..io_utils import save_vols

# global setup
this_file = os.path.basename(os.path.abspath(__file__)).split('.')[0]
OUTPUT_DIR = "/tmp/%s" % this_file


def test_get_slice_indices_ascending():
    np.testing.assert_array_equal(
        get_slice_indices(5, slice_order="ascending", return_final=True),
        [0, 1, 2, 3, 4])


def test_get_slice_indices_ascending_interleaved():
    np.testing.assert_array_equal(
        get_slice_indices(5, slice_order="ascending", interleaved=True,
                          return_final=True), [0, 3, 1, 4, 2])


def test_get_slice_indices_descending():
    # descending
    np.testing.assert_array_equal(
        get_slice_indices(5, slice_order="descending", return_final=True),
        [4, 3, 2, 1, 0])


def test_get_slice_indices_descending_interleaved():
    # descending and interleaved
    np.testing.assert_array_equal(
        get_slice_indices(5, slice_order="descending", interleaved=True,
                          return_final=True), [4, 1, 3, 0, 2])


def test_get_slice_indices_explicit():
    slice_order = [1, 4, 3, 2, 0]
    np.testing.assert_array_equal(
        get_slice_indices(5, slice_order=slice_order, return_final=True),
        [4, 0, 3, 2, 1])


def test_get_slice_indices_explicit_interleaved():
    slice_order = [1, 4, 3, 2, 0]
    with pytest.raises(ValueError):
        np.testing.assert_array_equal(
            get_slice_indices(5, slice_order=slice_order,
                              interleaved=True), [2, 0, 4, 1, 3])


def test_STC_constructor():
    stc = STC()
    assert stc.ref_slice == 0
    assert stc.interleaved == False
    assert stc.verbose == 1


def test_fMRISTC_constructor():
    fmristc = fMRISTC()
    assert fmristc.ref_slice == 0
    assert fmristc.interleaved == False
    assert fmristc.verbose == 1


def check_STC(true_signal, corrected_signal, ref_slice=0,
              rtol=None, atol=None):
    n_slices = true_signal.shape[2]
    np.testing.assert_array_almost_equal(
        corrected_signal[..., ref_slice, :],
        true_signal[..., ref_slice, :])
    for _ in range(1, n_slices):
        # relative closeness
        if rtol is not None:
            np.testing.assert_allclose(true_signal[..., 1:-1],
                                       corrected_signal[..., 1:-1],
                                       rtol=rtol)

        # relative closeness
        if atol is not None:
            np.testing.assert_allclose(true_signal[..., 1:-1],
                                       corrected_signal[..., 1:-1],
                                       atol=atol)


def test_STC_for_sinusoidal_mixture():
    # setup
    n_slices = 10
    n_rows = 3
    n_columns = 2
    slice_indices = np.arange(n_slices, dtype=int)
    timescale = .01
    sine_freq = [.5, .8, .11, .7]

    def my_sinusoid(t):
        """Creates mixture of sinusoids with different frequencies

        """

        res = t * 0

        for f in sine_freq:
            res += np.sin(2 * np.pi * t * f)

        return res

    time = np.arange(0, 24 + timescale, timescale)
    # signal = my_sinusoid(time)

    # define timing vars
    freq = 10
    TR = freq * timescale

    # sample the time
    acquisition_time = time[::freq]

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_indices * slice_TR
    shifted_acquisition_time = np.array([tau + acquisition_time
                                     for tau in time_shift])

    # acquire the signal at the corrupt sampled time points
    acquired_signal = np.array([
            [[my_sinusoid(shifted_acquisition_time[j])
              for j in range(n_slices)]
             for _ in range(n_columns)] for _ in range(n_rows)])

    n_scans = len(acquisition_time)

    # do STC
    stc = STC()
    stc.fit(n_slices=n_slices, n_scans=n_scans)
    stc.transform(acquired_signal)

    # truth
    true_signal = np.array([
        [[my_sinusoid(acquisition_time)
          for j in range(n_slices)] for _ in range(n_columns)]
        for _ in range(n_rows)])

    # check
    check_STC(true_signal, stc.output_data_, rtol=1.)
    check_STC(true_signal, stc.output_data_, atol=.13)


def test_STC_for_HRF():
    # setup
    import math
    n_slices = 10
    n_rows = 2
    n_columns = 3
    slice_indices = np.arange(n_slices, dtype=int)

    # create time values scaled at 1%
    timescale = .01
    n_timepoints = 24
    time = np.linspace(0, n_timepoints, num=int(1 + (n_timepoints - 0) / timescale))

    # create gamma functions
    n1 = 4
    lambda1 = 2
    n2 = 7
    lambda2 = 2
    a = .3
    c1 = 1
    c2 = .5

    def _compute_hrf(t):
        """Auxiliary function to compute HRF at given times (t)

        """

        hx = (t ** (n1 - 1)) * np.exp(
            -t / lambda1) / ((lambda1 ** n1) * math.factorial(n1 - 1))
        hy = (t ** (n2 - 1)) * np.exp(
            -t / lambda2) / ((lambda2 ** n2) * math.factorial(n2 - 1))

        # create hrf = weighted difference of two gammas
        hrf = a * (c1 * hx - c2 * hy)

        return hrf

    # sample the time and the signal
    freq = 100
    TR = 3.
    acquisition_time = time[::int(TR * freq)]
    n_scans = len(acquisition_time)

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_indices * slice_TR
    shifted_acquisition_time = np.array([tau + acquisition_time
                                     for tau in time_shift])

    # acquire the signal at the corrupt sampled time points
    acquired_sample = np.array([_compute_hrf(
                shifted_acquisition_time[j])
                                for j in range(n_slices)])
    acquired_sample = np.array([acquired_sample, ] * n_columns)
    acquired_sample = np.array([acquired_sample, ] * n_rows)

    # do STC
    stc = STC()
    stc.fit(n_scans=n_scans, n_slices=n_slices)
    stc.transform(acquired_sample)

    # truth
    true_signal = np.array([
            [[_compute_hrf(acquisition_time)
              for j in range(n_slices)]
             for _ in range(n_columns)] for _ in range(n_rows)])

    # check
    check_STC(true_signal, stc.output_data_, atol=.005)


def test_transform():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    film = nibabel.Nifti1Image(np.random.rand(11, 13, 17, 19),
                               np.eye(4))
    threeD_vols = nibabel.four_to_three(film)

    # filenames
    film_filename = os.path.join(output_dir, 'film.nii.gz')
    threeD_vols_filenames = [os.path.join(output_dir, 'fMETHODS-%06i.nii' % i)
                             for i in range(len(threeD_vols))]

    for stuff in [film, threeD_vols]:
        for as_files in [False, True]:
            if as_files:
                if isinstance(stuff, list):
                    basenames = [os.path.basename(x)
                                 for x in threeD_vols_filenames]
                else:
                    basenames = os.path.basename(film_filename)
                stuff = save_vols(stuff, output_dir, basenames=basenames)
            fmristc = fMRISTC().fit(raw_data=stuff)
            output = fmristc.transform(output_dir=output_dir)

            # test output type, shape, etc.
            if isinstance(stuff, list):
                assert isinstance(output, list)
                assert len(output) == film.shape[-1]
                if as_files:
                    assert os.path.basename(output[7]) == 'afMETHODS-000007.nii'
            else:
                if as_files:
                    assert os.path.basename(output) == 'afilm.nii.gz'


def test_get_slice_indices_not_final():
    # regression test for issue #232: by default, let backend software SPM,
    # decide the final order of indices of slices
    np.testing.assert_array_equal(
        get_slice_indices(5, slice_order="ascending", interleaved=True),
        [0, 2, 4, 1, 3])
