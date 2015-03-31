"""
XXX only use nosetests command-line tool to run this test module!

"""

import os
import sys
import numpy as np
import nibabel
import nose
import nose.tools
from numpy.testing import assert_array_almost_equal

# import the APIs to be tested
from ..time_diff import time_slice_diffs


def test_ts_diff_ana_null():
    # create basic L pattern
    n_scans = 3
    shape = (7, 8, 9, n_scans)
    film = np.zeros(shape)
    film[-3:, 5:-1, ..., ...] = 1
    film[..., 2:5, ..., ...] = 1
    scal = np.sum(film) * 1. / film.size
    affine = np.eye(4)
    film = nibabel.Nifti1Image(film, affine)
    nibabel.save(film, '/tmp/plop.nii')

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
    # create basic L pattern
    n_scans = 4
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
