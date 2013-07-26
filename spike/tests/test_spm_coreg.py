import numpy as np
import nose
import nose.tools
import numpy.testing


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
    vg = _tpvd_interp(g.ravel(), gshape, [0.], [2.], [1.2])
    nose.tools.assert_equal(vg, 63.2)
