import sys
import os
import numpy as np
import nose
import nose.tools
import numpy.testing
import scipy.io
import nibabel

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.dirname(
        os.path.split(os.path.abspath(__file__))[0]))
sys.path.append(PYPREPROCESS_DIR)

# import APIS to test
from spike.spm_coreg import(
    optfun
    )

def test_optfun():
    # setup
    decimal_precision = 6

    # affine matrix
    M = np.array([[3., 0., 0., 32.],
                  [0., 3., 0., 72.],
                  [0., 0., 1., 2.],
                  [0., 0., 0., 1.]])

    # load real-data
    tmp = scipy.io.loadmat(os.path.join(PYPREPROCESS_DIR,
                                        "test_data/spm_hist2_args_2.mat"),
                           squeeze_me=True, struct_as_record=False)

    VG, VFk = [tmp[k] for k in ["VG", "VFk"]]
    VG = nibabel.Nifti1Image(VG.uint8, VG.mat)
    VFk = nibabel.Nifti1Image(VFk.uint8, VFk.mat)

    # test NMI
    nose.tools.assert_almost_equal(optfun(np.zeros(6), VG, VFk,
                                          s=[1, 1, 1], cf='nmi'),
                                   -1.01313539922,
                                   places=decimal_precision)
    nose.tools.assert_almost_equal(optfun(np.zeros(6), VG, VFk,
                                          s=[4, 4, 4], cf='nmi'),
                                   -1.027121010234970,
                                   places=decimal_precision)

    # test MI
    nose.tools.assert_almost_equal(optfun(np.zeros(6), VG, VFk,
                                          s=[1, 1, 1], cf='mi'),
                                   -0.155388375906582,
                                   places=decimal_precision)
    nose.tools.assert_almost_equal(optfun(np.zeros(6), VG, VFk,
                                          s=[4, 4, 4], cf='mi'),
                                   -.316839496034499,
                                   places=decimal_precision)    
