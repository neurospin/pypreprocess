"""
XXX only use nosetests command-line tool to run this test module!

"""

import numpy as np
import nibabel
import nose
import nose.tools

# import the APIs to be tested
from ..spm_reslice import (
    reslice_vols
    )
from ..spm_realign import (
    _apply_realignment,
    )
from ..affine_transformations import (
    get_initial_motion_params
    )


def test_reslice_vols():
    # create basic L pattern
    n_scans = 3
    film = np.zeros((10, 10, 10, n_scans))
    film[-3:, 5:-1, ..., ...] = 1
    film[..., 2:5, ..., ...] = 1
    affine = np.array(  # XXX affine is fake!
        [[-2.99256921e+00, -1.12436414e-01, -2.23214120e-01,
           1.01544670e+02],
         [-5.69147766e-02, 2.87465930e+00, -1.07026458e+00,
            -8.77408752e+01],
         [-2.03200281e-01, 8.50703299e-01, 3.58708930e+00,
            -7.10269012e+01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            1.00000000e+00]])
    film = nibabel.Nifti1Image(film, affine)

    # rigidly move other volumes w.r.t. the first
    rp = np.array([get_initial_motion_params()
                   for _ in xrange(n_scans)])
    for t in xrange(film.shape[-1]):
        rp[t, ...][:3] += t / n_scans
        rp[t, ...][3:6] += np.pi * t

    film = list(_apply_realignment(film, rp))

    # affines are not the same
    nose.tools.assert_false(np.all(
            film[1].get_affine() == film[0].get_affine()))

    # reslice vols
    film = list(reslice_vols(film))

    # affines are now the same
    np.testing.assert_array_equal(film[1].get_affine(),
                                  film[0].get_affine())

# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
