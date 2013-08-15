import os
import sys
import numpy as np
import nibabel
import numpy.testing
import nose
import nose.tools
import inspect

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.split(os.path.abspath(__file__))[0])))
sys.path.append(PYPREPROCESS_DIR)

# import APIs to be tested
from algorithms.slice_timing.spm_slice_timing import (
    STC,
    fMRISTC,
    get_slice_indices,
    _load_fmri_data
    )
from external.nilearn.datasets import (
    fetch_spm_auditory_data
    )
from coreutils.io_utils import (
    save_vols
    )

# global setup
this_file = os.path.basename(os.path.abspath(__file__)).split('.')[0]
OUTPUT_DIR = "/tmp/%s" % this_file


def test_get_slice_indices_ascending():
    numpy.testing.assert_array_equal(
        get_slice_indices(5, slice_order="ascending"), [0, 1, 2, 3, 4])


def test_get_slice_indices_ascending_interleaved():
    numpy.testing.assert_array_equal(
        get_slice_indices(5, slice_order="ascending",
                          interleaved=True), [0, 3, 1, 4, 2])


def test_get_slice_indices_descending():
    # descending
    numpy.testing.assert_array_equal(
        get_slice_indices(5, slice_order="descending"), [4, 3, 2, 1, 0])


def test_get_slice_indices_descending_interleaved():
    # descending and interleaved
    numpy.testing.assert_array_equal(
        get_slice_indices(5, slice_order="descending",
                          interleaved=True), [4, 1, 3, 0, 2])


def test_get_slice_indices_explicit():
    slice_order = [1, 4, 3, 2, 0]
    numpy.testing.assert_array_equal(
        get_slice_indices(5, slice_order=slice_order), [4, 0, 3, 2, 1])


@nose.tools.raises(ValueError)
def test_get_slice_indices_explicit_interleaved():
    slice_order = [1, 4, 3, 2, 0]
    numpy.testing.assert_array_equal(
        get_slice_indices(5, slice_order=slice_order,
                          interleaved=True), [2, 0, 4, 1, 3])


def test_load_fmri_data_from_lists():
    raw_data = [[[[1., 0.],
                  [0., 1.]],

                 [[1., 0.],
                  [0., 1.]]],

                [[[1., 0.],
                  [0., 1.]],

                 [[1., 0.],
                  [0., 1.]]]]

    numpy.testing.assert_array_equal(_load_fmri_data(raw_data), raw_data)


def test_load_fmri_data_from_ndarray():
    raw_data = np.ndarray((3, 5, 7))
    numpy.testing.assert_array_equal(_load_fmri_data(raw_data), raw_data)

    raw_data = np.ndarray((3, 5, 7, 1))
    numpy.testing.assert_array_equal(_load_fmri_data(raw_data), raw_data)
    numpy.testing.assert_array_equal(_load_fmri_data(raw_data, is_3D=True),
                                     raw_data[..., 0])


def test_load_fmri_data_from_single_filename():
    data_path = os.path.join(
        os.environ["HOME"],
        ".nipy/tests/data/s12069_swaloc1_corr.nii.gz")
    if not os.path.exists(data_path):
        raise RuntimeError("You don't have nipy test data installed!")

    numpy.testing.assert_array_equal(_load_fmri_data(data_path).shape,
                                     (53, 63, 46, 128))


@nose.tools.nottest
def test_load_fmri_data_from_several_filenames():
    # fetch data
    spm_auditory_data = fetch_spm_auditory_data('/tmp')

    numpy.testing.assert_array_equal(
        _load_fmri_data(spm_auditory_data.func).shape, (64, 64, 64, 96))


def test_STC_constructor():
    stc = STC()

    nose.tools.assert_equal(stc.ref_slice, 0)
    nose.tools.assert_equal(stc.interleaved, False)
    nose.tools.assert_true(stc.verbose == 1)


def test_fMRISTC_constructor():
    fmristc = fMRISTC()

    nose.tools.assert_equal(fmristc.ref_slice, 0)
    nose.tools.assert_equal(fmristc.interleaved, False)
    nose.tools.assert_true(fmristc.verbose == 1)


def check_STC(true_signal, corrected_signal, ref_slice=0,
              rtol=None, atol=None):
    n_slices = true_signal.shape[2]

    numpy.testing.assert_array_almost_equal(
        corrected_signal[..., ref_slice, ...],
        true_signal[..., ref_slice, ...])

    for z in xrange(1, n_slices):
        # relative closeness
        if not rtol is None:
            numpy.testing.assert_allclose(true_signal[..., 1:-1],
                                          corrected_signal[..., 1:-1],
                                          rtol=rtol)

        # relative closeness
        if not atol is None:
            numpy.testing.assert_allclose(true_signal[..., 1:-1],
                                          corrected_signal[..., 1:-1],
                                          atol=atol)


def test_STC_for_sinusoidal_mixture(
                          ):
    # setup
    n_slices = 10
    n_rows = 3
    n_columns = 2
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
              for j in xrange(n_slices)]
             for y in xrange(n_columns)] for x in xrange(n_rows)]
                               )

    n_scans = len(acquisition_time)

    # do STC
    stc = STC()
    stc.fit(n_slices=n_slices, n_scans=n_scans)
    stc.transform(acquired_signal)

    # truth
    true_signal = np.array([
            [[my_sinusoid(acquisition_time)
              for j in xrange(n_slices)]
             for y in xrange(n_columns)] for x in xrange(n_rows)]
                               )

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
    time = np.linspace(0, n_timepoints, num=1 + (n_timepoints - 0) / timescale)

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
    acquisition_time = time[::TR * freq]
    n_scans = len(acquisition_time)

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_indices * slice_TR
    shifted_acquisition_time = np.array([tau + acquisition_time
                                     for tau in time_shift])

    # acquire the signal at the corrupt sampled time points
    acquired_sample = np.array([_compute_hrf(
                shifted_acquisition_time[j])
                                for j in xrange(n_slices)])
    acquired_sample = np.array([acquired_sample, ] * n_columns)
    acquired_sample = np.array([acquired_sample, ] * n_rows)

    # do STC
    stc = STC()
    stc.fit(n_scans=n_scans, n_slices=n_slices)
    stc.transform(acquired_sample)

    # truth
    true_signal = np.array([
            [[_compute_hrf(acquisition_time)
              for j in xrange(n_slices)]
             for y in xrange(n_columns)] for x in xrange(n_rows)]
                               )

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
    threeD_vols_filenames = [os.path.join(output_dir, 'fMETHODS-%06i' % i)
                             for i in xrange(len(threeD_vols))]

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
                nose.tools.assert_true(isinstance(
                        output, list))
                nose.tools.assert_equal(len(output),
                                        film.shape[-1])

                if as_files:
                    nose.tools.assert_equal(os.path.basename(output[7]),
                                            'afMETHODS-000007.nii.gz')
            else:
                if as_files:
                    nose.tools.assert_equal(os.path.basename(output),
                                            'afilm.nii.gz')                    


# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
