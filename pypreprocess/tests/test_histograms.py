import numpy as np
import nibabel
import numpy.testing
from ..histograms import(_correct_voxel_samp,
                         make_sampled_grid,
                         trilinear_interp,
                         joint_histogram
                         )


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

    assert trilinear_interp(f, shape, 1, 1, 1) == 0.
    assert trilinear_interp(f, shape, 2, 1, 1) == 1.
    assert trilinear_interp(f, shape, 1 + shape[0], 1, 1) == shape[0]
    assert 0. < trilinear_interp(f, shape, 1.5, 1., 1.) < 1.


def test_joint_histogram():
    ref_shape = (23, 29, 61)
    src_shape = (13, 51, 19)
    ref = np.arange(np.prod(ref_shape)).reshape(ref_shape)
    src = np.arange(np.prod(src_shape)).reshape(src_shape)

    # pre-sampled ref
    grid = make_sampled_grid(ref_shape, samp=2.)
    sampled_ref = trilinear_interp(ref.ravel(order='F'), ref_shape, *grid)
    jh = joint_histogram(sampled_ref, src, grid=grid, M=np.eye(4))
    assert jh.shape == (256, 256)
    assert np.all(jh >= 0)

    # ref not presampled
    jh = joint_histogram(nibabel.Nifti1Image(ref, np.eye(4)),
                         src, samp=np.pi, M=np.eye(4))
    assert jh.shape == (256, 256)
    assert np.all(jh >= 0)

    return jh
