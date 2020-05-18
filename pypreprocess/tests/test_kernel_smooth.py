import numpy as np
import nibabel
import numpy.testing
from ..kernel_smooth import fwhm2sigma, sigma2fwhm, smooth_image


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
        shape = rng.random_integers(20, size=ndim)
    ndim = len(shape)
    ndim = len(shape)
    if not n_scans is None and ndim == 4:
        shape[-1] = n_scans
    return parent_class(np.random.randn(*shape), affine)


def test_fwhm2sigma():
    fwhm = [1, 2, 3]
    for _fwhm in fwhm:
        numpy.testing.assert_array_equal(
            fwhm2sigma(_fwhm), np.array(_fwhm) / np.sqrt(8. * np.log(2)))
    for j in range(3):
        _fwhm = fwhm[j:]
        numpy.testing.assert_array_equal(
            fwhm2sigma(_fwhm), np.array(_fwhm) / np.sqrt(8. * np.log(2)))


def test_sigma2sigma():
    sigma = [7, 2, 3]
    for _sigma in sigma:
        numpy.testing.assert_array_equal(sigma2fwhm(_sigma),
                                         np.array(
                                             _sigma) * np.sqrt(8. * np.log(2)))
    for j in range(3):
        _sigma = sigma[j:]
        numpy.testing.assert_array_equal(sigma2fwhm(_sigma),
                                         np.array(
                                             _sigma) * np.sqrt(8. * np.log(2)))


def test_fwhm2sigma_and_sigma2fwhm_are_inverses():
    toto = [5, 7, 11.]
    numpy.testing.assert_array_equal(toto, sigma2fwhm(fwhm2sigma(toto)))
    numpy.testing.assert_array_almost_equal(toto, fwhm2sigma(sigma2fwhm(toto)))


def test_smooth_image_for_3D_vol():
    vol = create_random_image()
    svol = smooth_image(vol, [5, 7, 11.])
    assert svol.shape == vol.shape
    numpy.testing.assert_array_equal(svol.get_affine(), vol.get_affine())


def test_smooth_image_for_4D_film():
    film = create_random_image(ndim=4)
    sfilm = smooth_image(film, [5, 7., 11])
    assert sfilm.shape == film.shape
    numpy.testing.assert_array_equal(sfilm.get_affine(), film.get_affine())
