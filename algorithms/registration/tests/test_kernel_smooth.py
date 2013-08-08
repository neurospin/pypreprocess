"""
XXX only use nosetests command-line tool to run this test module!

"""

import os
import sys
import numpy as np
import nibabel
import scipy.io
import nose
import nose.tools
import numpy.testing

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.split(os.path.abspath(__file__))[0])))
sys.path.append(PYPREPROCESS_DIR)

# import the APIs to be tested
from coreutils.io_utils import _load_specific_vol
from algorithms.registration.kernel_smooth import (
    smooth_image
    )


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


def test_smooth_image_for_3D_vol():
    vol = create_random_image()

    svol = smooth_image(vol, [5, 7, 11.])

    nose.tools.assert_equal(svol.shape, vol.shape)

    numpy.testing.assert_array_equal(svol.get_affine(), vol.get_affine())


def test_smooth_image_for_4D_film():
    film = create_random_image(ndim=4)

    sfilm = smooth_image(film, [5, 7., 11])

    nose.tools.assert_equal(sfilm.shape, film.shape)

    numpy.testing.assert_array_equal(sfilm.get_affine(), film.get_affine())

# run all tests
if __name__ == "__main__":
    nose.runmodule(config=nose.config.Config(
            verbose=2,
            nocapture=True,
            ))
