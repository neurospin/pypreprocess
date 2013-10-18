"""
XXX only use nosetests command-line tool to run this test module!

"""

import numpy as np
import nose.tools

# import the APIs to be tested
from ..pypreprocess.affine_transformations import (
    get_initial_motion_params,
    spm_matrix,
    spm_imatrix,
    transform_coords,
    apply_realignment
    )
from ._test_utils import create_random_image


def test_get_initial_motion_params():
    # params for zero motion
    p = get_initial_motion_params()

    np.testing.assert_array_equal(p, [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])


def test_spm_matrix():
    p = get_initial_motion_params()

    # identity
    np.testing.assert_array_equal(spm_matrix(p),
                                  np.eye(4))

    # induce translations
    p[:3] += [.2, .3, .4]

    M = np.array([[1., 0., 0., .2],
                  [0., 1., 0., .3],
                  [0., 0., 1., .4],
                  [0., 0., 0., 1.]])

    np.testing.assert_array_equal(M, spm_matrix(p))


def test_spm_imatrix():
    p = get_initial_motion_params()

    # spm_matrix and spm_imatrix should be inverses of one another
    np.testing.assert_array_equal(spm_imatrix(spm_matrix(p)), p)


def test_transform_coords():
    p = get_initial_motion_params()
    M1 = np.eye(4)
    M2 = np.eye(4)
    coords = (0., 0., 1.)

    # rigidly move the voxel
    new_coords = transform_coords(p, M1, M2, coords)

    # coords shouldn't change
    nose.tools.assert_equal(new_coords.shape, (3, 1))
    np.testing.assert_array_equal(new_coords.ravel(), coords)


def test_apply_realignment_3D_niimg():
    # create 3D niimg
    vol = create_random_image(shape=(7, 11, 13))

    # apply realignment to vol
    rvol = apply_realignment(vol, [1, 2, 3, 4, 5, 6])

# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
