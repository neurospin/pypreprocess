"""

"""

import numpy as np
import nibabel
from numpy.testing import assert_array_almost_equal
from ..time_diff import (time_slice_diffs, multi_session_time_slice_diffs,
                         plot_tsdiffs)


def make_test_data(n_scans=3):
    shape = (7, 8, 9, n_scans)
    film = np.zeros(shape)
    film[-3:, 5:-1, ..., ...] = 1
    film[..., 2:5, ..., ...] = 1
    scal = np.sum(film) * 1. / film.size
    affine = np.eye(4)
    return nibabel.Nifti1Image(film, affine), scal


def test_ts_diff_ana_null():
    """ Run ts_diff_ana on constant image sequence """
    # create basic L pattern
    n_scans = 2
    film, scal = make_test_data(n_scans=n_scans)
    film.to_filename('/tmp/plop.nii')
    shape = film.shape
    report = time_slice_diffs(film)

    assert_array_almost_equal(report['volume_means'],
                              scal * np.ones(n_scans))
    assert_array_almost_equal(report['volume_mean_diff2'],
                              np.zeros(n_scans - 1))
    assert_array_almost_equal(report['slice_mean_diff2'],
                              np.zeros((n_scans - 1, shape[2])))
    assert_array_almost_equal(report['diff2_mean_vol'].get_data(),
                              np.zeros(shape[:3]))
    assert_array_almost_equal(report['slice_diff2_max_vol'].get_data(),
                              np.zeros(shape[:3]))


def test_ts_diff_ana():
    """ Run ts_diff_ana on changing image sequence """
    # create basic L pattern
    n_scans = 2
    shape = (7, 8, 9, n_scans)
    film = np.zeros(shape)
    film[-3:, 5:-1, ..., ...] = 1
    film[..., 2:5, ..., ...] = 1
    ref = film.copy()
    scal = np.sum(ref) * 1. / ref.size
    film = np.dot(film, np.diag(np.arange(n_scans)))

    affine = np.eye(4)
    film = nibabel.Nifti1Image(film, affine)

    report = time_slice_diffs(film)

    assert_array_almost_equal(report['volume_means'],
                              scal * np.arange(n_scans))
    assert_array_almost_equal(report['volume_mean_diff2'],
                              scal * np.ones(n_scans - 1))
    assert_array_almost_equal(report['slice_mean_diff2'],
                              scal * np.ones((n_scans - 1, shape[2]))
                              )
    assert_array_almost_equal(report['diff2_mean_vol'].get_data(),
                              ref[..., 0])
    assert_array_almost_equal(report['slice_diff2_max_vol'].get_data(),
                              ref[..., 0])


def test_ts_diff_ana_two_session():
    """ Run ts_diff_ana on two image sequences """
    # create basic L pattern
    n_scans = 2
    film, _ = make_test_data(n_scans=n_scans)
    film, affine = film.get_data(), film.get_affine()
    shape = film.shape
    ref = film.copy()
    scal = np.sum(ref) * 1. / ref.size
    film = np.dot(film, np.diag(np.arange(n_scans)))

    affine = np.eye(4)
    films = []
    for _ in range(2):
        films.append(nibabel.Nifti1Image(film, affine))

    report = multi_session_time_slice_diffs(films)

    assert_array_almost_equal(report['volume_means'],
                              np.tile(scal * np.arange(n_scans), 2))
    assert_array_almost_equal(report['volume_mean_diff2'],
                              np.tile(scal * np.ones(n_scans - 1), 2))
    assert_array_almost_equal(
        report['slice_mean_diff2'],
        np.tile(scal * np.ones((n_scans - 1, shape[2])), (2, 1)))
    assert_array_almost_equal(report['diff2_mean_vol'].get_data(),
                              ref[..., 0])
    assert_array_almost_equal(report['slice_diff2_max_vol'].get_data(),
                              ref[..., 0])


def test_plot_tsdiffs_no_crash():
    n_scans, n_sessions = 2, 2
    rng = np.random.RandomState(42)
    shape = (7, 8, 9, n_scans)
    films = []
    affine = np.eye(4)
    for _ in range(n_sessions):
        film = np.zeros(shape)
        msk = (rng.randn(*shape) > .7)
        film[msk] = rng.randn(msk.sum())
        films.append(nibabel.Nifti1Image(film, affine))

    results = multi_session_time_slice_diffs(films)
    for use_same_figure in [True, False]:
        plot_tsdiffs(results, use_same_figure=use_same_figure)
