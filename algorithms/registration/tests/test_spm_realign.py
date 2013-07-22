"""
XXX only use nosetests command-line tool to run this test module!

"""

import os
import sys
import numpy as np
import nibabel
import nose
import nose.tools

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.split(os.path.abspath(__file__))[0])))
sys.path.append(PYPREPROCESS_DIR)

# import the APIs to be tested
from coreutils.io_utils import _load_specific_vol
from algorithms.registration.spm_realign import (
    _compute_rate_of_change_of_chisq,
    _apply_realignment,
    _extract_realignment_params,
    # MRIMotionCorrection
    )
from algorithms.registration.affine_transformations import (
    get_initial_motion_params)


def create_random_image(shape=None,
                        ndim=3,
                        n_scans=None,
                        affine=np.eye(4),
                        parent_class=nibabel.Nifti1Image):
    """
    Creates a random image of prescribed shape

    """

    if shape is None:
        shape = np.random.random_integers(20, size=ndim)

    ndim = len(shape)
    if not n_scans is None and ndim == 4:
        shape[-1] = n_scans

    return parent_class(np.random.randn(*shape), affine)


def test_compute_rate_of_change_of_chisq():
    # setup
    true_A = np.array(
        [[-0., -0., -0., -0., -0., -0.],
         [1., -0., -0., -0., 1.000001, 1.000001],
         [1., -0., -0., -0., 1.0000015, 1.0000015],
         [-2., 1., -0., 1.000001, -2.000001, -5.],
         [1., -0., -0., -0., 1.000001, 2.000001],
         [1., -0., -0., -0., 1.0000015, 2.0000015],
         [-2., 1., -0., 1.0000015, -2.000001, -6.9999995],
         [1., -0., -0., -0., 1.000001, 3.000001],
         [1., -0., -0., -0., 1.0000015, 3.0000015],
         [-2., 1., -0., 1.000002, -2.000001, -8.999999],
         [1., -0., -0., -0., 1.000001, 4.000001],
         [1., -0., -0., -0., 1.0000015, 4.0000015],
         [-2., -3., 1., -7.0000005, -5., 0.9999975],
         [1., -0., -0., -0., 2.000001, 1.000001],
         [1., -0., -0., -0., 2.0000015, 1.0000015],
         [-2., 1., -0., 2.000001, -4.000001, -5.],
         [1., -0., -0., -0., 2.000001, 2.000001],
         [1., -0., -0., -0., 2.0000015, 2.0000015],
         [-2., 1., -0., 2.0000015, -4.000001, -6.9999995],
         [1., -0., -0., -0., 2.000001, 3.000001],
         [1., -0., -0., -0., 2.0000015, 3.0000015],
         [-2., 1., -0., 2.000002, -4.000001, -8.999999],
         [1., -0., -0., -0., 2.000001, 4.000001],
         [1., -0., -0., -0., 2.0000015, 4.0000015],
         [-2., -3., 1., -10., -6.9999995, 0.9999975],
         [1., -0., -0., -0., 3.000001, 1.000001],
         [1., -0., -0., -0., 3.0000015, 1.0000015],
         [-2., 1., -0., 3.000001, -6.000001, -5.],
         [1., -0., -0., -0., 3.000001, 2.000001],
         [1., -0., -0., -0., 3.0000015, 2.0000015],
         [-2., 1., -0., 3.0000015, -6.000001, -6.9999995],
         [1., -0., -0., -0., 3.000001, 3.000001],
         [1., -0., -0., -0., 3.0000015, 3.0000015],
         [-2., 1., -0., 3.000002, -6.000001, -8.999999],
         [1., -0., -0., -0., 3.000001, 4.000001],
         [1., -0., -0., -0., 3.0000015, 4.0000015],
         [-2., -3., 1., -12.9999995, -8.999999, 0.9999975],
         [1., -0., -0., -0., 4.000001, 1.000001],
         [1., -0., -0., -0., 4.0000015, 1.0000015],
         [-2., 1., -0., 4.000001, -8.000001, -5.],
         [1., -0., -0., -0., 4.000001, 2.000001],
         [1., -0., -0., -0., 4.0000015, 2.0000015],
         [-2., 1., -0., 4.0000015, -8.000001, -6.9999995],
         [1., -0., -0., -0., 4.000001, 3.000001],
         [1., -0., -0., -0., 4.0000015, 3.0000015],
         [-2., 1., -0., 4.000002, -8.000001, -8.999999],
         [1., -0., -0., -0., 4.000001, 4.000001],
         [1., -0., -0., -0., 4.0000015, 4.0000015],
         [-2., -3., 1., -15.999999, -10.9999985, 0.9999975],
         [1., -0., -0., -0., 5.000001, 1.000001],
         [1., -0., -0., -0., 5.0000015, 1.0000015],
         [-2., 1., -0., 5.000001, -10.000001, -5.],
         [1., -0., -0., -0., 5.000001, 2.000001],
         [1., -0., -0., -0., 5.0000015, 2.0000015],
         [-2., 1., -0., 5.0000015, -10.000001, -6.9999995],
         [1., -0., -0., -0., 5.000001, 3.000001],
         [1., -0., -0., -0., 5.0000015, 3.0000015],
         [-2., 1., -0., 5.000002, -10.000001, -8.999999],
         [1., -0., -0., -0., 5.000001, 4.000001],
         [1., -0., -0., -0., 5.0000015, 4.0000015]])
    decimal_precision = 8  # precision for array comparison
    lkp = [0, 1, 2, 3, 4, 5]  # translations + rotations model
    grid = np.mgrid[1:4:, 1:5:, 1:6:].reshape((3, -1),
                                              # true_A came from matlab
                                              order='F')
    gradG = np.vstack(([0, 0, 0], np.diff(grid, axis=1).T)).T  # image gradient
    M = np.eye(4)  # grid transformation matrix

    # compute A
    A = _compute_rate_of_change_of_chisq(M, grid, gradG, lkp=lkp)

    # compare A with true_A (from spm implementation)
    np.testing.assert_array_almost_equal(A, true_A, decimal=decimal_precision)


def test_appy_realigment_and_extract_realignment_params_APIs():
    # create data
    n_scans = 10
    affine  = np.eye(4)
    affine[:3, -1] = [96, -96, -69]
    film = create_random_image(shape=[64, 64, 64, n_scans], affine=affine)

    # there should be no motion
    for t in xrange(n_scans):
        np.testing.assert_array_equal(_extract_realignment_params(
                _load_specific_vol(film, t)[0],
                _load_specific_vol(film, 0)[0]),
                                      get_initial_motion_params())

    # now introduce motion into other vols relative to the first vol
    rp = np.ndarray((n_scans, 12))
    for t in xrange(n_scans):
        rp[t, ...] = get_initial_motion_params()

        if t > 0:
            rp[t, :3] += [t, t + 1, t + 2]  # translation

    # apply motion (noise)
    film = list(_apply_realignment(film, rp, inverse=False))

    # check that motion has been induced
    for t in xrange(n_scans):
        _tmp = get_initial_motion_params()

        if t > 0:
            _tmp[:3] += [-t, -1 * (t + 1), -1 * (t + 2)]  # translation

        np.testing.assert_array_equal(_extract_realignment_params(
                _load_specific_vol(film, t)[0],
                _load_specific_vol(film, 0)[0]),
                                      _tmp)

# def test_MRIMotionCorrection():
#     # create basic L pattern
#     n_scans = 3
#     film = np.zeros((10, 10, 10, n_scans))
#     film[-3:, 5:-1, ..., ...] = 1
#     film[..., 2:5, ..., ...] = 1
#     affine = np.array(
#         [[ -2.99256921e+00,  -1.12436414e-01,  -2.23214120e-01,
#             1.01544670e+02],
#          [ -5.69147766e-02,   2.87465930e+00,  -1.07026458e+00,
#             -8.77408752e+01],
#          [ -2.03200281e-01,   8.50703299e-01,   3.58708930e+00,
#             -7.10269012e+01],
#          [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#             1.00000000e+00]])
#     film = nibabel.Nifti1Image(film, affine)

#     # rigidly move other volumes w.r.t. the first
#     rp = np.array([get_initial_motion_params()
#                    for _ in xrange(n_scans)])
#     for t in xrange(film.shape[-1]):
#         rp[t, ...][:3] += t / n_scans
#         rp[t, ...][3:6] += np.pi * t

#     film = [_apply_realignment(film, rp)]

#     # pack film into nifti image object
#     # XXX the following affine is utterly fake!!!

#     # instantiate object
#     mrimc = MRIMotionCorrection(quality=1., lkp=[0, 1, 2])

#     # estimate motion
#     mrimc.fit(film)

#     # reslice vols to absorb estimated motion
#     mrimc.transform('/tmp/toto', reslice=True)

#     # realignment params for ref vol should imply no motion
#     np.testing.assert_array_equal(
#         mrimc._rp[0][0],
#         [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.])

# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
