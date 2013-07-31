import sys
import os
import numpy as np
import nose
import nose.tools
import numpy.testing
import scipy.io
import commands

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.dirname(
        os.path.split(os.path.abspath(__file__))[0]))
sys.path.append(PYPREPROCESS_DIR)

# import APIS to test
# XXX the auto-installation hack below is not here to stay!!!
try:
    from spike.spm_hist2py import(
        samppy,
        hist2py
        )
except ImportError:
    print "spm_hist2.so module not yet built; I'll try to build it for ye."
    print commands.getoutput(
        "make -C %s" % os.path.dirname(os.path.split(
                os.path.abspath(__file__))[0]))

    from spike.spm_hist2py import(
        samppy,
        hist2py
        )


def test_tpvd_interp():
    # create dummy image
    gshape = (64, 64, 64)
    g = np.ndarray(gshape, dtype='uint8')
    for i, j, k in np.ndindex(gshape):
        g[i, j, k] = i + j + k

    # do tpvd (trilinear partial volume distribution) interpolation
    vg = samppy(g, 0., 2., 1.2)
    nose.tools.assert_almost_equal(vg, 63.20000076)


def test_hist2py():
    ###
    # SPM's spm_hist2 is the ground truth
    ###

    # affine matrix
    M = np.array([[3., 0., 0., 32.],
                  [0., 3., 0., 72.],
                  [0., 0., 1., 2.],
                  [0., 0., 0., 1.]])

    # load real-data
    tmp = scipy.io.loadmat(os.path.join(PYPREPROCESS_DIR,
                                        "test_data/spm_hist2_args_1.mat"),
                           squeeze_me=True, struct_as_record=False)

    VG, VFk = [tmp[k] for k in ["VG", "VFk"]]

    # compute joint histogram
    H = hist2py(M, VG.uint8, VFk.uint8, [4, 4, 4])
    h = H.ravel(order='F')

    # test the histogram (SPM's is the ground-truth
    nose.tools.assert_true(np.all(h) >= 0)
    nose.tools.assert_almost_equal(h.mean(), 0.04388427734375, places=13)
    nose.tools.assert_equal(h[h > 0].sum(), 2876)
    nose.tools.assert_equal((h > 0).sum(), 3106)
    numpy.testing.assert_almost_equal(h[h > 0][5:10],
                                      [1.615593433380127, 0.595069885253906,
                                       1.995013236999512, 1.409916877746582,
                                       0.536985397338867])
