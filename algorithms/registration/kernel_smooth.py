"""
:Module: kernel_smooth
:Synopsis: assorted utilities for smoothing images (to absorb noise before
computing a gradient, etc.). A good starting point for understanding this code
is the smooth_image(..) wrapper function
:Author: DOHMATOB Elvis Dopgima, adapted from nipy source code.
Credits to nipy dev.

"""

import numpy as np
import numpy.fft as npfft
import nibabel as ni
import scipy.linalg
import gc
import affine_transformations


def fwhm2sigma(fwhm):
    """Convert a FWHM value to sigma in a Gaussian kernel.

    Parameters
    ----------
    fwhm : array-like
       FWHM value or values

    Returns
    -------
    sigma : array or float
       sigma values corresponding to `fwhm` values

    Examples
    --------
    >>> sigma = fwhm2sigma(6)
    >>> sigmae = fwhm2sigma([6, 7, 8])
    >>> sigma == sigmae[0]
    True

    """

    fwhm = np.asarray(fwhm)

    return fwhm / np.sqrt(8 * np.log(2))


def sigma2fwhm(sigma):
    """Convert a sigma in a Gaussian kernel to a FWHM value.

    Parameters
    ----------
    sigma : array-like
       sigma value or values

    Returns
    -------
    fwhm : array or float
       fwhm values corresponding to `sigma` values

    Examples
    --------
    >>> fwhm = sigma2fwhm(3)
    >>> fwhms = sigma2fwhm([3, 4, 5])
    >>> fwhm == fwhms[0]
    True
    """
    sigma = np.asarray(sigma)
    return sigma * np.sqrt(8 * np.log(2))


def _crop(X, tol=1.0e-10):
    """Find a bounding box for support of fabs(X) > tol and returned
    crop region.

    """

    aX = np.fabs(X)
    n = len(X.shape)
    I = np.indices(X.shape)[:, np.greater(aX, tol)]
    if I.shape[1] > 0:
        m = [I[i].min() for i in range(n)]
        M = [I[i].max() for i in range(n)]
        slices = [slice(m[i], M[i] + 1, 1) for i in range(n)]
        return X[slices]
    else:
        return np.zeros((1, ) * n)


class LinearFilter(object):
    """A class to implement some FFT smoothers for Image objects.
    By default, this does a Gaussian kernel smooth. More choices
    would be better!

    """

    normalization = 'l1sum'

    def __init__(self, affine, shape, fwhm=6.0, scale=1.0, location=0.0,
                 cov=None):
        """
        Parameters
        ----------
        affine : 2D array of shape (4, 4)
        shape : sequence
        fwhm : float, optional
           fwhm for Gaussian kernel, default is 6.0
        scale : float, optional
           scaling to apply to data after smooth, default 1.0
        location : float
           offset to apply to data after smooth and scaling, default 0
        cov : None or array, optional
           Covariance matrix

        """

        self._affine = affine
        self._ndims = tuple([len(shape)] * 2)
        self._bshape = shape
        self._fwhm = fwhm
        self._scale = scale
        self._location = location
        self._cov = cov
        self._setup_kernel()

    def _setup_kernel(self):
        # voxel indices of array implied by shape
        voxels = np.indices(self._bshape).astype(np.float64)

        # coordinates of physical center.  XXX - why the 'floor' here?
        vox_center = np.floor((np.array(self._bshape) - 1) / 2.0)
        phys_center = affine_transformations.get_physical_coords(self._affine,
                                                                 vox_center)

        # reshape to (N coordinates, -1).  We appear to need to assign
        # to shape instead of doing a reshape, in order to avoid memory
        # copies
        voxels.shape = (voxels.shape[0], np.product(voxels.shape[1:]))

        # physical coordinates relative to center
        X = (affine_transformations.get_physical_coords(self._affine,
                                                        voxels.T) - phys_center
             ).T
        X.shape = (self._ndims[1],) + tuple(self._bshape)

        # compute kernel from these positions
        kernel = self(X, axis=0)
        kernel = _crop(kernel)
        self.norms = {'l2': np.sqrt((kernel ** 2).sum()),
                      'l1': np.fabs(kernel).sum(),
                      'l1sum': kernel.sum()}
        self._kernel = kernel
        self._shape = (np.ceil((np.asarray(self._bshape) +
                              np.asarray(kernel.shape)) / 2) * 2 + 2)
        self.fkernel = np.zeros(self._shape)
        slices = [slice(0, kernel.shape[i]) for i in range(len(kernel.shape))]
        self.fkernel[slices] = kernel
        self.fkernel = npfft.rfftn(self.fkernel)

        return kernel

    def _normsq(self, X, axis=-1):
        """
        Compute the (periodic, i.e. on a torus) squared distance needed for
        FFT smoothing. Assumes coordinate system is linear.

        Parameters
        ----------
        X : array
           array of points
        axis : int, optional
           axis containing coordinates. Default -1

        """

        # copy X
        _X = np.array(X)

        # roll coordinate axis to front
        _X = np.rollaxis(_X, axis)

        # convert coordinates to FWHM units
        if self._fwhm is not 1.0:
            f = fwhm2sigma(self._fwhm)
            if f.shape == ():
                f = np.ones(len(self._bshape)) * f
            for i in range(len(self._bshape)):
                _X[i] /= f[i]

        # whiten ?
        if self._cov != None:
            _chol = scipy.linalg.cholesky(self._cov)
            _X = np.dot(scipy.linalg.inv(_chol), _X)
        # compute squared distance
        D2 = np.sum(_X ** 2, axis=0)

        return D2

    def __call__(self, X, axis=-1):
        """Compute kernel from points

        Parameters
        ----------
        X : array
           array of points
        axis : int, optional
           axis containing coordinates.  Default -1

        """

        _normsq = self._normsq(X, axis) / 2.
        t = np.less_equal(_normsq, 15)

        return np.exp(-np.minimum(_normsq, 15)) * t

    def smooth(self, in_data, clean=False, is_fft=False):
        """Apply smoothing to `inimage`

        Parameters
        ----------
        in_data : array_like
           The array to be smoothed. should be same shape as the
           shape provided during instantiation of this object
        clean : bool, optional
           Should we call ``nan_to_num`` on the data before smoothing?
        is_fft : bool, optional
           Has the data already been fft'd?

        Returns
        -------
        s_image : `Image`
           New image, with smoothing applied

        """

        ndim = len(in_data.shape)

        if ndim == 4:
            _out = np.ndarray(in_data.shape)
            n_scans = in_data.shape[-1]
        elif ndim == 3:
            n_scans = 1
        else:
            raise NotImplementedError('expecting either 3 or 4-d image')

        for _scan in range(n_scans):
            if ndim == 4:
                data = in_data[..., _scan]
            elif ndim == 3:
                data = in_data[:]
            if clean:
                data = np.nan_to_num(data)
            if not is_fft:
                data = self._presmooth(data)
            data *= self.fkernel
            data = npfft.irfftn(data) / self.norms[self.normalization]
            gc.collect()
            _dslice = [slice(0, self._bshape[i], 1) for i in range(3)]
            if self._scale != 1:
                data = self._scale * data[_dslice]
            if self._location != 0.0:
                data += self._location
            gc.collect()
            # Write out data
            if ndim == 4:
                _out[..., _scan] = data
            else:
                _out = data
            _scan += 1

        gc.collect()
        _out = _out[[slice(self._kernel.shape[i] / 2, self._bshape[i] +
                           self._kernel.shape[i] / 2)
                     for i in range(len(self._bshape))]]

        # return output
        return _out

    def _presmooth(self, indata):
        slices = [slice(0, self._bshape[i], 1)
                  for i in range(len(self._shape))]
        _buffer = np.zeros(self._shape)
        _buffer[slices] = indata

        return npfft.rfftn(_buffer)


def smooth_image(img, fwhm, **kwargs):
    """Function wrapper LinearFilter class. Spatially smoothens img with
    kernel of size fwhm.

    Parameters
    ----------
    img: ``ni.Nifti1Image``
        image to be smoothen
    fwhm: 1D array like of size as big as there are spatial dimensions
    in the image
        FWHM of smoothing kernel
    **kwargs: dict-like
        key-word arguments passed to LinearFilter constructor.

    Returns
    -------
    Smoothened image, same type and size as the input img.

    """

    smoothing_kernel = LinearFilter(
        img.get_affine(),
        img.shape,
        fwhm=fwhm,
        **kwargs)

    return ni.Nifti1Image(smoothing_kernel.smooth(img.get_data(),
                                                  clean=True),
                          img.get_affine())
