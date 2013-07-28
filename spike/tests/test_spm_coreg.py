import sys
import os
import numpy as np
import nose
import nose.tools
import numpy.testing
import scipy.io

THIS_DIR = os.path.split(os.path.abspath(__file__))[0]

# import APIS to test
from ..spm_coreg import(
    _tpvd_interp,
    _joint_histogram
    )


def test_tpvd_interp():
    # create dummy image
    gshape = (64, 64, 64)
    g = np.ndarray(gshape, dtype='uint8')
    for i, j, k in np.ndindex(gshape):
        g[i, j, k] = i + j + k

    # do tpvd (trilinear partial volume distribution) interpolation
    vg = _tpvd_interp(g.ravel(), g.shape, [0.], [2.], [1.2])
    nose.tools.assert_equal(vg, 63.2)


def test_joint_histogram():
    ###
    # SPM's spm_hist2 is the ground truth
    ###
    # create dummy ref image
    gshape = (64, 64, 64)
    g = np.ndarray(gshape, dtype='uint8')
    for i, j, k in np.ndindex(gshape):
        g[i, j, k] = i + j + k

    # create dummy src image
    fshape = (256, 256, 54)
    f = np.ndarray(fshape, dtype='uint8')
    for i, j, k in np.ndindex(fshape):
        f[i, j, k] = i + j + k

    # affine matrix
    M = np.array([[3., 0., 0., 32.],
                  [0., 3., 0., 72.],
                  [0., 0., 1., 2.],
                  [0., 0., 0., 1.]])

    # compute joint histogram
    H = _joint_histogram(g.ravel(), f.ravel(), M=M, gshape=g.shape,
                         fshape=f.shape,
                         s=[4, 4, 4])
    h = H.ravel(order='F')

    # test the histogram
    nose.tools.assert_true(np.all(h) >= 0)
    nose.tools.assert_almost_equal(h.mean(), 0.043884, places=6)
    nose.tools.assert_almost_equal(h[h > 0].mean(), 0.574052, places=6)
    nose.tools.assert_true((h > 0).sum(), 2876)
    numpy.testing.assert_array_almost_equal(h[h > 0][-7:],
                                            [0.982986, 0.790970, 0.209030,
                                             0.327209, 0.672791, 0.413254,
                                             0.586746],
                                            decimal=2)
    nose.tools.assert_almost_equal(h.max(), 2.643997, places=2)

    # load real-data
    tmp = scipy.io.loadmat(os.path.join(THIS_DIR, "spm_hist2_args.mat"),
                           squeeze_me=True,
                           struct_as_record=False)
    VG, VFk = [tmp[k] for k in ["VG", "VFk"]]

    # compute joint histogram
    H = _joint_histogram(VG.uint8.ravel(order='F'),
                         VFk.uint8.ravel(order='F'),
                         M=M, gshape=VG.dim, fshape=VFk.dim,
                         s=[4, 4, 4])
    h = H.ravel(order='F')

    # test the histogram
    nose.tools.assert_true(np.all(h) >= 0)
    nose.tools.assert_almost_equal(h.mean(), 0.04388427734375, places=13)
    nose.tools.assert_true(np.abs(h.sum() - 2876) < 1e-4)
