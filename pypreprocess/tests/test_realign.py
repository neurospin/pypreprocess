"""
XXX only use nosetests command-line tool to run this test module!

"""

import os
import inspect
import numpy as np
import nibabel
import scipy.io
import nose
import nose.tools
import numpy.testing

# import the APIs to be tested
from ..io_utils import load_specific_vol
from ..realign import (
    _compute_rate_of_change_of_chisq,
    _apply_realignment,
    _extract_realignment_params,
    MRIMotionCorrection
    )
from ..affine_transformations import (
    get_initial_motion_params)

# global setup
THIS_FILE = os.path.abspath(__file__).split('.')[0]
THIS_DIR = os.path.dirname(THIS_FILE)
OUTPUT_DIR = "/tmp/%s" % os.path.basename(THIS_FILE)


def create_random_image(shape=None,
                        ndim=3,
                        n_scans=None,
                        affine=np.eye(4),
                        parent_class=nibabel.Nifti1Image):
    """
    Creates a random image of prescribed shape

    """

    rng = np.random.RandomState(0)

    if shape is None:
        shape = np.random.random_integers(20, size=ndim)

    ndim = len(shape)

    ndim = len(shape)
    if not n_scans is None and ndim == 4:
        shape[-1] = n_scans

    return parent_class(np.random.randn(*shape), affine)


def _make_vol_specific_translation(translation, n_scans, t):
    """
    translation: constant part plus vol-dependent part

    """

    return (t > 0) * (translation + 1. * t / n_scans)


def _make_vol_specific_rotation(rotation, n_scans, t):
    """
    rotation: constant part plus vol-dependent part

    """

    return (t > 0) * (rotation + 1. * t / n_scans
                      ) * np.pi / 180.


def test_compute_rate_of_change_of_chisq():
    # XXX please document strange variables !!!
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
    decimal_precision = 8  # precision for array comparison (SPM is grnd-truth)
    lkp = [0, 1, 2, 3, 4, 5]  # translations + rotations model
    grid = np.mgrid[1:4:, 1:5:, 1:6:].reshape((3, -1),
                                              # true_A came from matlab
                                              order='F')
    gradG = np.vstack(([0, 0, 0], np.diff(grid, axis=1).T)).T  # image gradient
    M = np.eye(4)  # grid transformation matrix

    # compute A
    A = _compute_rate_of_change_of_chisq(M, grid, gradG, lkp=lkp)

    # compare A with true_A (from spm implementation)
    numpy.testing.assert_array_almost_equal(A, true_A,
                                            decimal=decimal_precision)


def test_appy_realigment_and_extract_realignment_params_APIs():
    # setu
    n_scans = 10
    translation = np.array([1, 2, 3])  # mm
    rotation = np.array([3, 2, 1])  # degrees

    # create data
    affine = np.array([[-3., 0., 0., 96.],
                       [0., 3., 0., -96.],
                       [0., 0., 3., -69.],
                       [0., 0., 0., 1.]])
    film = create_random_image(shape=[16, 16, 16, n_scans], affine=affine)

    # there should be no motion
    for t in xrange(n_scans):
        numpy.testing.assert_array_equal(_extract_realignment_params(
                load_specific_vol(film, t)[0],
                load_specific_vol(film, 0)[0]),
                                      get_initial_motion_params())

    # now introduce motion into other vols relative to the first vol
    rp = np.ndarray((n_scans, 12))
    for t in xrange(n_scans):
        rp[t, ...] = get_initial_motion_params()
        rp[t, :3] += _make_vol_specific_translation(translation, n_scans, t)
        rp[t, 3:6] += _make_vol_specific_rotation(rotation, n_scans, t)

    # apply motion (noise)
    film = list(_apply_realignment(film, rp))

    # check that motion has been induced
    for t in xrange(n_scans):
        _tmp = get_initial_motion_params()
        _tmp[:3] += _make_vol_specific_translation(translation, n_scans, t)
        _tmp[3:6] += _make_vol_specific_rotation(rotation, n_scans, t)

        numpy.testing.assert_array_almost_equal(_extract_realignment_params(
                load_specific_vol(film, t)[0],
                load_specific_vol(film, 0)[0]),
                                             _tmp)


def test_MRIMotionCorrection_fit():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_scans = 3
    lkp = np.arange(6)
    translation = np.array([1, 3, 2])  # mm
    rotation = np.array([1, 2, .5])  # degrees
    MAX_RE = .12  # we'll test for this max relative error in estimating motion

    # create data
    vol = scipy.io.loadmat(os.path.join(THIS_DIR,
                                        "test_data/spmmmfmri.mat"),
                           squeeze_me=True, struct_as_record=False)
    data = np.ndarray(list(vol['data'].shape) + [n_scans])
    for t in xrange(n_scans):
        data[..., t] = vol['data']
    film = nibabel.Nifti1Image(data, vol['affine'])

    # rigidly move other volumes w.r.t. the first
    rp = np.array([get_initial_motion_params()
                   for _ in xrange(n_scans)])
    for t in xrange(n_scans):
        rp[t, ...][:3] += _make_vol_specific_translation(
            translation, n_scans, t)
        rp[t, ...][3:6] += _make_vol_specific_rotation(rotation, n_scans, t)

    film = list(_apply_realignment(film, rp))

    # instantiate object
    mrimc = MRIMotionCorrection(quality=1., lkp=lkp).fit([film])

    # check shape of realignment params
    numpy.testing.assert_array_equal(mrimc.rp_.shape, [1] + [n_scans, 6])

    # check that we estimated the correct motion params
    # XXX refine the notion of "closeness" below
    for t in xrange(n_scans):
        _tmp = get_initial_motion_params()[:6]

        # check the estimated motion is well within our MAX_RE limit
        _tmp[:3] += _make_vol_specific_translation(translation, n_scans, t)
        _tmp[3:6] += _make_vol_specific_rotation(rotation, n_scans, t)
        if t > 0:
            numpy.testing.assert_allclose(mrimc.rp_[0][t][lkp],
                                          _tmp[lkp], rtol=MAX_RE)
        else:
            numpy.testing.assert_array_equal(mrimc.rp_[0][t],
                                             get_initial_motion_params()[:6])

    ####################
    # check transform
    ####################
    mrimc_output = mrimc.transform(output_dir)

    nose.tools.assert_true(mrimc_output['realigned_files'], basestring)

# run all tests
if __name__ == "__main__":
    nose.runmodule(config=nose.config.Config(
            verbose=2,
            nocapture=True,
            ))
