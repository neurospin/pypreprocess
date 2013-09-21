import os
import numpy as np
import nose
import nose.tools
import numpy.testing
import nibabel
import scipy.io

# import APIS to test
from ..pypreprocess.coreg import(
    _correct_voxel_samp,
    make_sampled_grid,
    mask_grid,
    trilinear_interp,
    joint_histogram,
    compute_similarity_from_jhist,
    Coregister
    )
from ..pypreprocess.affine_transformations import apply_realignment_to_vol

# global setup
THIS_FILE = os.path.abspath(__file__).split('.')[0]
THIS_DIR = os.path.dirname(THIS_FILE)
OUTPUT_DIR = "/tmp/%s" % os.path.basename(THIS_FILE)


def test_correct_voxel_samp():
    numpy.testing.assert_array_equal(
        _correct_voxel_samp(np.eye(4), 2), [2., 2., 2.])

    numpy.testing.assert_array_equal(
        _correct_voxel_samp(np.eye(4), [3, 2, 1]), [3., 2., 1.])

    numpy.testing.assert_array_equal(_correct_voxel_samp(
            np.array([[-1., 0., 0., 128.],
                      [0., 1., 0., -168.],
                      [0., 0., 3., -75.],
                      [0., 0., 0., 1.]]),
            4), [4., 4., 4. / 3])


def test_make_sampled_grid_without_spm_magic():
    for samp in [1., [1.], [1.] * 3]:
        numpy.testing.assert_array_equal(make_sampled_grid([3, 5, 7],
                                                           samp=samp,
                                                           magic=False),
                                         np.array(
                [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3.],
                 [1., 2., 3., 4., 5., 1., 2., 3., 4., 5., 1., 2., 3., 4., 5.]
                 ]))


def test_trilinear_interp():
    shape = (23, 29, 31)
    f = np.arange(np.prod(shape))

    nose.tools.assert_equal(trilinear_interp(f, shape, 1, 1, 1), 0.)
    nose.tools.assert_equal(trilinear_interp(f, shape, 2, 1, 1), 1.)
    nose.tools.assert_equal(trilinear_interp(f, shape, 1 + shape[0], 1, 1),
                            shape[0])
    nose.tools.assert_true(0. < trilinear_interp(f, shape, 1.5, 1., 1.) < 1.)


def test_joint_histogram():
    ref_shape = (23, 29, 61)
    src_shape = (13, 51, 19)
    ref = np.arange(np.prod(ref_shape)).reshape(ref_shape)
    src = np.arange(np.prod(src_shape)).reshape(src_shape)

    # pre-sampled ref
    grid = make_sampled_grid(ref_shape, samp=2.)
    sampled_ref = trilinear_interp(ref.ravel(order='F'), ref_shape, *grid)
    jh = joint_histogram(sampled_ref, src, grid=grid, M=np.eye(4))
    nose.tools.assert_equal(jh.shape, (256, 256))
    nose.tools.assert_true(np.all(jh >= 0))

    # ref not presampled
    jh = joint_histogram(nibabel.Nifti1Image(ref, np.eye(4)),
                         src, samp=np.pi, M=np.eye(4))
    nose.tools.assert_equal(jh.shape, (256, 256))
    nose.tools.assert_true(np.all(jh >= 0))

    return jh


def test_compute_similarity_from_jhist():
    jh = test_joint_histogram()

    for cost_fun in ['mi', 'nmi', 'ecc']:
        s = compute_similarity_from_jhist(jh, cost_fun=cost_fun)
        nose.tools.assert_true(s <= --1)


def test_coregister_on_toy_data():
    shape = (23, 29, 31)
    ref = nibabel.Nifti1Image(np.arange(np.prod(shape)).reshape(shape),
                              np.eye(4)
                              )

    # rigidly move reference vol to get a new volume: the source vol
    src = apply_realignment_to_vol(ref, [1, 1, 1,  # translations
                                         0, .01, 0,  # rotations
                                         ])

    # learn realignment params for coregistration: src -> ref
    c = Coregister(sep=[4, 2, 1]).fit(ref, src)

    # compare estimated realigment parameters with ground-truth
    numpy.testing.assert_almost_equal(-c.params_[4], .01, decimal=2)
    numpy.testing.assert_array_almost_equal(-c.params_[[3, 5]],
                                             [0, 0], decimal=2)
    numpy.testing.assert_array_equal(np.round(-c.params_)[[0, 1, 2]],
                                     [1., 1., 1.])


def test_coregister_on_real_data():
    # load data
    _tmp = scipy.io.loadmat(
        os.path.join(THIS_DIR, "test_data/some_anat.mat"),
        squeeze_me=True, struct_as_record=False)
    ref = nibabel.Nifti1Image(_tmp['data'], _tmp['affine'])

    # rigidly move reference vol to get a new volume: the source vol
    src = apply_realignment_to_vol(ref, [1, 2, 3,  # translations
                                         0, .01, 0,  # rotations
                                         ])

    # learn realignment params for coregistration: src -> ref
    c = Coregister().fit(ref, src)

    # compare estimated realigment parameters with ground-truth
    numpy.testing.assert_almost_equal(-c.params_[4], .01, decimal=4)
    numpy.testing.assert_array_almost_equal(-c.params_[[3, 5]],
                                             [0, 0], decimal=4)
    numpy.testing.assert_array_equal(np.round(-c.params_)[[0, 1, 2]],
                                     [1., 2., 3.])

# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
