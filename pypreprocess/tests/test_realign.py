import os
import inspect
import numpy as np
import scipy.io
import pytest
import nibabel
from nibabel.processing import smooth_image as nibabel_smoothing
from nilearn.image import index_img
from ..realign import _compute_rate_of_change_of_chisq, MRIMotionCorrection
from ..affine_transformations import (
    apply_realignment, extract_realignment_params, get_initial_motion_params)
from ._test_utils import create_random_image

# global setup
THIS_FILE = os.path.abspath(__file__).split('.')[0]
THIS_DIR = os.path.dirname(THIS_FILE)
OUTPUT_DIR = "/tmp/%s" % os.path.basename(THIS_FILE)


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
    np.testing.assert_array_almost_equal(A, true_A,
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
    for t in range(n_scans):
        np.testing.assert_array_equal(
            extract_realignment_params(index_img(film, t), index_img(film, 0)),
            get_initial_motion_params())

    # now introduce motion into other vols relative to the first vol
    rp = np.ndarray((n_scans, 12))
    for t in range(n_scans):
        rp[t, ...] = get_initial_motion_params()
        rp[t, :3] += _make_vol_specific_translation(translation, n_scans, t)
        rp[t, 3:6] += _make_vol_specific_rotation(rotation, n_scans, t)

    # apply motion (noise)
    film = apply_realignment(film, rp)

    # check that motion has been induced
    for t in range(n_scans):
        _tmp = get_initial_motion_params()
        _tmp[:3] += _make_vol_specific_translation(translation, n_scans, t)
        _tmp[3:6] += _make_vol_specific_rotation(rotation, n_scans, t)

        np.testing.assert_array_almost_equal(
            extract_realignment_params(film[t], film[0]), _tmp)


def test_MRIMotionCorrection_fit():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    n_scans = 2
    lkp = np.arange(6)
    translation = np.array([1, 3, 2])  # mm
    rotation = np.array([1, 2, .5])  # degrees
    MAX_RE = .12  # we'll test for this max relative error in estimating motion

    # create data
    vol = scipy.io.loadmat(os.path.join(THIS_DIR,
                                        "test_data/spmmmfmri.mat"),
                           squeeze_me=True, struct_as_record=False)
    data = np.ndarray(list(vol['data'].shape) + [n_scans])
    for t in range(n_scans): data[..., t] = vol['data']
    film = nibabel.Nifti1Image(data, vol['affine'])

    # rigidly move other volumes w.r.t. the first
    rp = np.array([get_initial_motion_params() for _ in range(n_scans)])
    for t in range(n_scans):
        rp[t, ...][:3] += _make_vol_specific_translation(
            translation, n_scans, t)
        rp[t, ...][3:6] += _make_vol_specific_rotation(rotation, n_scans, t)

    film = apply_realignment(film, rp)

    _kwargs = {'quality': 1., 'lkp': lkp}
    for n_jobs in [1, 2]:
        for smooth_func in [None, nibabel_smoothing]:
            kwargs = _kwargs.copy()
            if smooth_func is not None:
                kwargs['smooth_func'] = smooth_func
            # instantiate object
            mrimc = MRIMotionCorrection(**kwargs).fit([film], n_jobs=n_jobs)

            # check shape of realignment params
            np.testing.assert_array_equal(np.array(
                mrimc.realignment_parameters_).shape, [1] + [n_scans, 6])

            # check that we estimated the correct motion params
            # XXX refine the notion of "closeness" below
            for t in range(n_scans):
                _tmp = get_initial_motion_params()[:6]

                # check the estimated motion is well within our MAX_RE limit
                _tmp[:3] += _make_vol_specific_translation(
                    translation, n_scans, t)
                _tmp[3:6] += _make_vol_specific_rotation(rotation, n_scans, t)
                if t > 0: np.testing.assert_allclose(
                        mrimc.realignment_parameters_[0][t][lkp],
                        _tmp[lkp], rtol=MAX_RE)
                else: np.testing.assert_array_equal(
                        mrimc.realignment_parameters_[0][t],
                        get_initial_motion_params()[:6])

            ####################
            # check transform
            ####################
            mrimc_output = mrimc.transform(output_dir)
            assert len(mrimc_output['realigned_images']) == 1
            assert len(set(mrimc_output['realigned_images'][0])) == n_scans
            assert len(set(mrimc_output['realigned_images'][0])) == n_scans

@pytest.mark.skip()
def test_bug_fix_issue_36_on_realign():
    from pypreprocess.datasets import fetch_spm_auditory
    sd = fetch_spm_auditory("/tmp/spm_auditory/")

    # shouldn't throw an IndexError
    MRIMotionCorrection(n_sessions=8, quality=1.).fit(
        [sd.func[:2], sd.func[:3]] * 4).transform("/tmp")
