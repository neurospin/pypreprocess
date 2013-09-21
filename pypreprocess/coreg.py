"""
:Module: coreg.py
:Synopsis: Histogram-based co-registration of 3D MRI images.
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

import scipy.ndimage
import scipy.optimize
import scipy.special
import numpy as np
import nibabel
from .affine_transformations import (spm_matrix,
                                     nibabel2spm_affine
                                     )
from .io_utils import (loaduint8,
                       is_niimg,
                       get_basenames,
                       save_vol,
                       load_specific_vol
                       )
from .kernel_smooth import fwhm2sigma

# 'texture' of floats in machine precision
EPS = np.finfo(float).eps

# magic table from SPM
ran = np.array([0.656619, 0.891183, 0.488144, 0.992646, 0.373326,
                0.531378, 0.181316, 0.501944, 0.422195,
                0.660427, 0.673653, 0.95733, 0.191866, 0.111216,
                0.565054, 0.969166, 0.0237439, 0.870216,
                0.0268766, 0.519529, 0.192291, 0.715689, 0.250673,
                0.933865, 0.137189, 0.521622, 0.895202,
                0.942387, 0.335083, 0.437364, 0.471156, 0.14931,
                0.135864, 0.532498, 0.725789, 0.398703,
                0.358419, 0.285279, 0.868635, 0.626413, 0.241172,
                0.978082, 0.640501, 0.229849, 0.681335,
                0.665823, 0.134718, 0.0224933, 0.262199, 0.116515,
                0.0693182, 0.85293, 0.180331, 0.0324186,
                0.733926, 0.536517, 0.27603, 0.368458, 0.0128863,
                0.889206, 0.866021, 0.254247, 0.569481,
                0.159265, 0.594364, 0.3311, 0.658613, 0.863634,
                0.567623, 0.980481, 0.791832, 0.152594,
                0.833027, 0.191863, 0.638987, 0.669, 0.772088,
                0.379818, 0.441585, 0.48306, 0.608106,
                0.175996, 0.00202556, 0.790224, 0.513609, 0.213229,
                0.10345, 0.157337, 0.407515, 0.407757,
                0.0526927, 0.941815, 0.149972, 0.384374, 0.311059,
                0.168534, 0.896648
                ])


def _correct_voxel_samp(affine, samp):
    """
    Corrects a sampling rate (samp) according to the image's resolution.

    Parameters
    ----------
    affine: 2D array of shape (4, 4)
        affine matrix of the underlying image

    samp: 1D array_like of 3 floats, optional (default [1, 1, 1])
        sampling rate (in millimeters) along each axis

    Returns
    -------
    samp_: array of 3 floats
        samp corrected according to the affine matrix of the underlying
        image

    """

    return 1. * np.array(samp) / np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def mask_grid(grid, shape):
    """
    Remove voxels that have fallen out of the FOV.

    Parameters
    ----------
    grid: 2D array of shape (3, n_points)
        the grid been masked

    shape: array of 3 ints
        the shape of the underlying image

    Returns
    -------
    msk: 1D array of n_points bools
        a msk for lattices/voxels/points on the grid that have
        not falled out of the underlying image's FOV

    """

    # sanitize input
    assert grid.ndim == 2
    assert grid.shape[0] == 3
    assert len(shape) == 3

    # create complement of mask
    msk = ((grid[0, ...] < 1) +
           (grid[0, ...] >= shape[0]) +
           (grid[1, ...] < 1) +
           (grid[1, ...] >= shape[1]) +
           (grid[2, ...] < 1) +
           (grid[2, ...] >= shape[2]))

    # return the mask
    return ~msk


def make_sampled_grid(shape, samp=[1., 1., 1.], magic=True):
    """
    Creates a regular meshgrid of given shape, sampled in each coordinate
    direction.

    Parameters
    ----------
    shape: 1D array_like of 3 integers
        shape of the image being sampled

    samp: 1D array_like of 3 floats, optional (default [1, 1, 1])
        sampling rate (in millimeters) along each axis

    magic: bool, optional (default True)
        if set, then the created grid will be doped with a strange list
        of values cooked by SPM people; strangely enough nothing works
        without this magic

    Returns
    --------
    grid: 2D array of shape (3, V / floor(V_0)), where V is the volume of the
    image being sampled and V_0 is the volume of the parallelopiped with sides
    samp.
        the sampled grid

    """

    # sanitize input
    assert len(shape) == 3, (
        "Expected triplet of 3 integers, got %s") % shape

    if len(np.shape(samp)) == 0:
        samp = [samp] * 3
    elif np.shape(samp) == (1,):
        samp = [samp[0]] * 3
    else:
        print samp

    samp = np.array(samp)

    assert samp.ndim == 1 and len(samp) == 3, (
        "samp must be float of triple of floats, got %s") % samp

    # build the grid
    if magic:
        iran = 0
        grid = []
        for z in np.arange(1., shape[2] - samp[2], samp[2]):
            for y in np.arange(1., shape[1] - samp[1], samp[1]):
                for x in np.arange(1., shape[0] - samp[0], samp[0]):

                    # doping
                    iran = (iran + 1) % 97
                    rx = x + ran[iran] * samp[0]
                    iran = (iran + 1) % 97
                    ry = y + ran[iran] * samp[1]
                    iran = (iran + 1) % 97
                    rz = z + ran[iran] * samp[2]

                    grid.append([rx, ry, rz])

        return np.array(grid).T
    else:
        return np.mgrid[1:shape[0] - samp[0]:samp[0],
                        1:shape[1] - samp[1]:samp[1],
                        1:shape[2] - samp[2]:samp[2]].reshape((3, -1))


def trilinear_interp(f, shape, x, y, z):
    """
    Performs trilinear interpolation to determine the value of a function
    f at the 3D point(s) with coordinates (x, y, z), generally not aligned
    with knots/vertices of the image's grid.

    Parameters
    ----------
    f: 1D array of floats
        image intensity values to be interpolated

    shape: list or tuple of 3 ints
        shape of the image f

    x: float or array_like of floats
       x coordinate(s) of the point(s)

    y: float or array_like of floats
       y coordinate(s) of the point(s)

    z: float or array_like of floats
       z coordinate(s) of the point(s)

    Returns
    -------
    vf: float if (x, y, z) is a single point, of a list of length n_points
    if (x, y, z) if a list of n_points (i.e each of x, y, and z is a list
    of n_points floats)
        the interpolated value of f at the point(s) (x, y, z)

    """

    # cast everything to floats to avoid abnoxious supprises
    f = np.asarray(f, dtype=np.float)

    # sanitiy checks
    assert f.ndim == 1
    assert len(shape) == 3

    # "differential geometry" along x-axis
    ix = np.floor(x)
    dx1 = x - ix
    dx2 = 1.0 - dx1

    # differential geometry along y-axis
    iy = np.floor(y)
    dy1 = y - iy
    dy2 = 1.0 - dy1

    # differential geometry along z-axis
    iz = np.floor(z)
    dz1 = z - iz
    dz2 = 1.0 - dz1

    # create table of offsets into the image
    offset = np.asarray(ix - 1 + shape[0] * (iy - 1 + shape[1] * (iz - 1)),
                        dtype=np.integer)

    # compute contribution of the corners of the parallelopiped (aka voxel,
    # aka cube) in which the point(s) (x, y, z) live(s)
    k222 = f[offset + 0]
    k122 = f[offset + 1]
    k212 = f[offset + shape[0]]
    k112 = f[offset + shape[0] + 1]
    offset = offset + shape[0] * shape[1]
    k221 = f[offset + 0]
    k121 = f[offset + 1]
    k211 = f[offset + shape[0]]
    k111 = f[offset + shape[0] + 1]

    # combine everything to get the value of f at the point(s) (x, y, z)
    return (((k222 * dx2 + k122 * dx1) * dy2  +\
                 (k212 * dx2 + k112 * dx1) * dy1)) * dz2 +\
                 (((k221 * dx2 + k121 * dx1) * dy2 +\
                       (k211 * dx2 + k111 * dx1) * dy1)) * dz1


def joint_histogram(ref, src, grid=None, samp=None, M=np.eye(4),
                    bins=(256, 256)):
    """
    Function to compute the joint histogram between of a reference and a source
    (moving) image, under an optioanl rigid transformation (M) of the source
    image.

    Parameters
    ----------

    ref: 1D array of length n_voxels
        the reference image already sampled accuring to the current sampling
        rate (in pyramidal/multi-resolution loop)

    src: 3D array of shape
        the moving image to be resampled unto the reference image's grid,
        deformed rigidly according to the matrix M

    grid: 2D array of shape (3, n_voxels), optional (defaul None)
        grid of reference image (ref). If a value is provided, then
        it is assumed that ref is the result of sampling the reference image
        on this case

    samp: 1D array_like of 3 floats, optional (default [1, 1, 1])
        sampling rate (in millimeters) along each axis. Exactly one of samp
        and grid must be provided. If a value if provided (i.e if grid is None)
        then the grid of the reference image (ref) is created and sampled
        in along each coordinate direction at the rate samp; then the reference
        image is resampled on this grid; then the grid is rigidly moved by M,
        and the source (moving) image is resampled thereupon. Note that,

    M: 2D array of shape (4, 4)
        affine transformation matrix for transforming the source (moving)
        image before computing the joint histogram. Thus the grid is rigidly
        moved under action by L, and then the source image (src) is resampled
        on this new grid

    bins: pair of integers, optional (default (256, 256))
       shape of the joint histogram being computed

    Returns
    -------
    jh: 2D array of shape bins
       the joint histogram of the reference imafe (ref) and the
       source/moving image (src), the latter subject to an affine
       transformation M

    """

    # sanity checks
    assert M.shape == (4, 4), M.shape
    assert src.ndim == 3, src.shape

    if grid is None:
        assert not samp is None, "Both grid and samp can't be None"
        assert is_niimg(ref), (
            "grid is None, expected niimg for ref, got %s"
            ) % type(ref)

        # create sampled grid for ref img
        grid = make_sampled_grid(ref.shape, samp=_correct_voxel_samp(
                ref.get_affine(), samp))

        # interpolate ref on sampled grid
        ref = trilinear_interp(ref.get_data().ravel(order='F'),
                               ref.shape, *grid)
    else:
        assert ref.ndim == 1, ref.shape
        assert grid.shape == (3, len(ref)), grid.shape
        assert isinstance(ref, np.ndarray)
        assert ref.ndim == 1

    # rigidly deform grid of reference image to obtain grid of moving image
    deformed_grid = np.dot(M, np.vstack((grid,
                                         np.ones(grid.shape[1]))))[:-1, ...]

    # mask out points that have fallen out of the source image's FOV
    msk = mask_grid(deformed_grid, src.shape)
    deformed_grid = deformed_grid[..., msk]
    _ref = ref[msk]

    warped_src = trilinear_interp(src.ravel(order='F'), src.shape,
                                  *deformed_grid)

    # compute joint histogram proper
    # XXX all the bottle neck is in the following call to numpy's histogram2d
    return np.histogram2d(_ref, warped_src, bins=bins)[0]


def _smoothing_kernel(fwhm, x):
    """
    Creates a gaussian kernel centered at x, and with given fwhm.

    """

    # variance from fwhm
    s = fwhm ** 2 / (8 * np.log(2)) + EPS

    # Gaussian convolve with 0th degree B-spline
    w1 = .5 * np.sqrt(2 / s)
    w2 = -.5 / s
    w3 = np.sqrt(s / 2 / np.pi)
    krn = .5 * (scipy.special.erf(w1 * (x + 1)) * (x + 1) + scipy.special.erf(
            w1 * (x - 1)) * (x - 1) - 2 * scipy.special.erf(
            w1 * x) * x) + w3 * (np.exp(w2 * (x + 1) ** 2) + np.exp(
            w2 * (x - 1) ** 2) - 2 * np.exp(w2 * x ** 2))

    krn[krn < 0.] = 0

    return krn


def compute_similarity_from_jhist(jh, fwhm=[7, 7], cost_fun='nmi'):
    """
    Computes an information-theoretic similarity from a joint histogram.

    Parameters
    ----------
    jh: 2D array
        joint histogram of the two random variables being compared

    fwhm: float or pair of float
        kernel width for smoothing the joint histogram

    cost_fun: string, optional (default 'nmi')
        smilarity model to use; possible values are:
        'mi': Mutual Information
        'nmi': Normalized Mutual Information
        'ecc': Entropic Cross-Correlation

    Returns
    -------
    o: float
        the computed similariy measure

    """

    # sanitize input
    assert len(np.shape(jh)) == 2, "jh must be 2D array, got %s" % jh

    if len(np.shape(fwhm)) == 0:
        fwhm = [fwhm] * 2

    fwhm = fwhm[:2]

    # smoothing the jh
    if 0x1:
        # create separable filter
        lim  = np.ceil(fwhm * 2)
        krn1 = _smoothing_kernel(fwhm[0], np.linspace(-1 * lim[0], lim[0],
                                                      num=2 * lim[0]))
        krn1 = krn1 / np.sum(krn1)
        krn2 = _smoothing_kernel(fwhm[1], np.linspace(-1 * lim[1], lim[1],
                                                      num=2 * lim[1]))
        krn2 = krn2 / np.sum(krn2)

        # smooth the histogram with kern1 x kern2
        jh = scipy.signal.sepfir2d(jh, krn1, krn2)
    else:
        # smooth the jh with a gaussian filter of given fwhm
        jh = scipy.ndimage.gaussian_filter(jh, sigma=fwhm2sigma(fwhm[:2]),
                                           mode='wrap'
                                           )

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # compute cost function proper
    if cost_fun == 'mi':
        # Mutual Information:
        jh = jh * np.log2(jh / np.dot(s2, s1))
        mi = np.sum(jh)
        o = -mi
    elif cost_fun == 'ecc':
        # Entropy Correlation Coefficient of:
        # Maes, Collignon, Vandermeulen, Marchal & Suetens (1997).
        # "Multimodality image registration by maximisation of mutual
        # information". IEEE Transactions on Medical Imaging 16(2):187-198
        jh = jh * np.log2(jh / np.dot(s2, s1))
        mi = np.sum(jh.ravel(order='F'))
        ecc = -2 * mi / (np.sum(s1 * np.log2(s1)) + np.sum(s2 * np.log2(s2)))
        o = -ecc
    elif cost_fun == 'nmi':
        # Normalised Mutual Information of:
        # Studholme,  jhill & jhawkes (1998).
        # "A normalized entropy measure of 3-D medical image alignment".
        # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
        nmi = (np.sum(s1 * np.log2(s1)) + np.sum(
                s2 * np.log2(s2))) / np.sum(np.sum(jh * np.log2(jh)))
        o = -nmi
    else:
        raise NotImplementedError(
            "Unsupported cost_fun (cost function): %s" % cost_fun)

    return o


def compute_similarity(x, ref, src, ref_affine, src_affine, grid,
                       cost_fun='nmi', fwhm=[7, 7], bins=(256, 256)):
    """
    Computes the similarity between the reference image (ref) and the moving
    image src, under the current affine motion parameters (x).

    The computed similarity measure is histogram-based.

    Parameters
    ----------
    x: 1D array of 6 floats
        current affine motion parameters; specifies an affine transformation
        matrix

    ref: 1D array of length n_voxels
        the reference image already sampled accuring to the current sampling
        rate (in pyramidal/multi-resolution loop)

    src: 3D array of shape
        moving image to be resampled unto the reference image's grid, deformed
        rigidly according to the matrix M

    ref_affine: 2D array of shape (4, 4)
        affine matrix of reference image (ref

    src_affine: 2D arfray of shape (4, 4)
        affine matrix of moving image (src)

    grid: 2D array of shape (3, n_voxels)
        grid of reference image (ref)

    bins: pair of integers, optional (default (256, 256))
       shape of the joint histogram being computed

    cost_fun: string, optional (default 'nmi')
        smilarity model to use; possible values are:
        'mi': Mutual Information
        'nmi': Normalized Mutual Information
        'ecc': Entropic Cross-Correlation

    fwhm: 1D array of 2 floats
        FWHM for smoothing the joint histogram

    Returns
    -------
    o: float
        the computed similariy measure

    """

    # compute affine transformation matrix
    x = np.array(x)
    M = np.dot(scipy.linalg.lstsq(src_affine,
                                  spm_matrix(x))[0],
                                  ref_affine)

    # create the joint histogram
    jh = joint_histogram(ref.copy(), src.get_data(), grid=grid, M=M, bins=bins)

    # compute similarity from joint histgram
    return compute_similarity_from_jhist(jh, fwhm=fwhm, cost_fun=cost_fun)


def _run_powell(x0, xi, tolsc, *otherargs):
    """
    Run Powell optimization.

    Parameters
    ----------
    x0: 1D array of 6 floats
        starting estimates for realignment parameters

    xi: 2D array of shape (n_directions, 6)
        search directions for Brent's line-search

    tolsc: 1D array of 6 floats
        absolute tolerance in each of the 6 realignment parameters
        (for numerical convergence criterion)

    *otherargs: tuple
        argments to be passed to the `compute_similarity` backend

    Returns
    -------
    x_opt: 1D array of 6 floats
        optimal realignment parameters found

    """

    def _compute_similarity(x):
        """
        Just a sandbox.

        """

        output = compute_similarity(x, *otherargs)

        # verbose
        token = "".join(['%-12.4g ' % z for z in x])
        token += '|  %.5g' % output
        print token

        return output

    # fire!
    return scipy.optimize.fmin_powell(_compute_similarity, x0,
                                      direc=xi,
                                      xtol=min(np.min(tolsc), 1e-3),
                                      )


class Coregister(object):
    """
    Similarity-based rigid-body multi-modal registration.

    Parameters
    ----------
    sep: 1D array of floats, optional (default [4, 2])
        pyramidal optimization seperation (in mm)

    params: 1D array of length 6, optional (default [0, 0, 0, 0, 0, 0]
        starting estimates


    tol: 1D array of 6 floats, optional (
    default [.02, .02, .02, .001, .001, .001])
        tolerances for the accuracy of each parameter

    cost_fun: string, optional (default "nmi")
        similarity function to be optimized. Possible values are:
        "mi": Mutual Information
        "nmi": Normalized Mutual Information
        "ecc": Entropy Correlation Coefficient

    fwhm: 1D array of 2 floats
        FWHM for smoothing the joint histogram

    bins: pair of integers, optional (default (256, 256))
       shape of the joint histogram being computed

    Attributes
    ----------
    params_: 1D array of 6 floats (3 translations + 3 rotations)
        the realign parameters estimated

    """

    def __init__(self,
                 sep=np.array([4, 2]),
                 params=np.zeros(6),
                 tol=np.array([.02, .02, .02, .001, .001, .001]),
                 cost_fun="nmi",
                 smooth_vols=True,
                 fwhm=np.array([7., 7., 7.]),
                 bins=(256, 256),
                 verbose=1
                 ):

        self.sep = np.array(sep)
        self.params = np.array(params)
        self.tol = np.array(tol)
        self.cost_fun = cost_fun
        self.fwhm = np.array(fwhm)
        self.bins = bins
        self.smooth_vols = smooth_vols
        self.verbose = verbose

        # configure the Powell optimizer
        self.sc = np.array(tol)
        self.sc = self.sc[:len(params)]
        self.xi = np.diag(self.sc * 20)

    def _log(self, msg):
        """Logs a message, according to verbose level.

        Parameters
        ----------
        msg: string
            message to log

        """

        if self.verbose:
            print(msg)

    def fit(self, ref, src):
        """
        Estimates the co-registration parameters for rigidly registering
        vol to vol, which can be of different modality
        (fMRI and PET, etc.).

        The technology used is histogram-based registration.

        Parameters
        ----------
        vol: string (existing filename) or 3D (if 4D then then first volume
        will be used) nibabel image object
            reference volume (image that is kept fixed)
        vol: string (existing filename) or 3D nibabel image object
            source image (image that is jiggled about to fit the
            reference image)

        Returns
        -------
        Returns
        -------
        `Coregistration` instance
            fitted object

        """

        # load vols
        ref = loaduint8(ref)
        src = loaduint8(src)

        # tweak affines so we can play SPM games everafter
        ref = nibabel.Nifti1Image(ref.get_data(),
                                  nibabel2spm_affine(ref.get_affine()))
        src = nibabel.Nifti1Image(src.get_data(),
                                  nibabel2spm_affine(src.get_affine()))

        # smooth images according to pyramidal sep
        if self.smooth_vols:
            # ref
            vxg = np.sqrt(np.sum(ref.get_affine()[:3, :3] ** 2, axis=0))
            fwhmg = np.sqrt(np.maximum(
                    np.ones(3) * self.sep[-1] ** 2 - vxg ** 2,
                    [0, 0, 0])) / vxg
            ref = nibabel.Nifti1Image(
                scipy.ndimage.gaussian_filter(ref.get_data(),
                                              fwhm2sigma(fwhmg)),
                ref.get_affine())

            # src
            vxf = np.sqrt(np.sum(src.get_affine()[:3, :3] ** 2, axis=0))
            fwhmf = np.sqrt(np.maximum(
                    np.ones(3) * self.sep[-1] ** 2 - vxf ** 2,
                    [0, 0, 0])) / vxf
            src = nibabel.Nifti1Image(scipy.ndimage.gaussian_filter(
                src.get_data(), fwhm2sigma(fwhmf)),
                                      src.get_affine())

        # pyramidal loop
        self.params_ = np.array(self.params)
        for samp in self.sep:
            print ("\r\nRunning Powell gradient-less local optimization "
                   "(pyramidal level = %smm)...") % samp

            # create sampled grid for ref img
            grid = make_sampled_grid(ref.shape, samp=_correct_voxel_samp(
                    ref.get_affine(), samp))

            # interpolate ref on sampled grid
            sampled_ref = trilinear_interp(ref.get_data().ravel(order='F'),
                                           ref.shape, *grid)

            # find optimal realignment parameters
            self.params_ = _run_powell(self.params_, self.xi, self.sc,
                                       sampled_ref, src, ref.get_affine(),
                                       src.get_affine(), grid, self.cost_fun,
                                       self.fwhm, self.bins)

        return self

    def transform(self,
                  src,
                  output_dir=None,
                  prefix="Coreg",
                  ext=".nii.gz",
                  ):
        """
        Applies estimated co-registration parameter to the input volume
        (src).

        Parameters
        ----------
        src: string (existing filename) or 3D nibabel image object
            source image (image that is jiggled about to fit the
            reference image)
        output_dir: string, optional (dafault None)
            existing dirname where output will be written
        prefix: string, optional (default 'Coreg')
            prefix for output filenames.
        ext: string, optional (default ".nii.gz")
            file extension for ouput images

        Returns
        -------
        coregistered_source: nibabel image object or existing filename
        (if output_dir is specified)
            the coregistered source volume

        """

        # load vol
        src = load_specific_vol(src, 0)[0]

        # apply coreg
        coregistered_source = nibabel.Nifti1Image(src.get_data(),
                                                  scipy.linalg.lstsq(
                spm_matrix(self.params_),
                src.get_affine())[0])

        # save output unto disk
        if not output_dir is None:
            if isinstance(src, basestring):
                basename = get_basenames(src)
            else:
                basename = 'coregistered_source'

            coregistered_source = save_vol(coregistered_source,
                                           output_dir=output_dir,
                                           basename=basename,
                                           ext=ext, prefix=prefix)

        return coregistered_source
