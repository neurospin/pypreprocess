"""
:Module: histograms
:Synopsis: building sampled meshgrids, masking meshgrids, performing trilinear
interpolation, computation of joint histograms between fixed and moving
3D images, etc.
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

import numpy as np
from .affine_transformations import get_physical_coords
from .io_utils import is_niimg


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
    return (((k222 * dx2 + k122 * dx1) * dy2  + \
                 (k212 * dx2 + k112 * dx1) * dy1)) * dz2 + \
                 (((k221 * dx2 + k121 * dx1) * dy2 + \
                       (k211 * dx2 + k111 * dx1) * dy1)) * dz1


def joint_histogram(ref, src, grid=None, samp=None, M=np.eye(4),
                    bins=(256, 256)):
    """
    Function to compute the joint histogram between a reference and a source
    (moving) image, under an optional rigid transformation (M) of the source
    image.

    Parameters
    ----------

    ref: 1D array of length n_voxels
        the reference image already sampled accuring to the current sampling
        rate (in pyramidal/multi-resolution loop). The ravelled order of ref
        is assumed to be 'F'.

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
                               ref.shape, grid[0], grid[1], grid[2])
    else:
        # ref image already sampled on coarse grid
        assert ref.ndim == 1, ref.shape
        assert grid.shape == (3, len(ref)), grid.shape
        assert isinstance(ref, np.ndarray)
        assert ref.ndim == 1

    # rigidly deform grid of reference image to obtain grid of moving image
    deformed_grid = get_physical_coords(M, grid)

    # mask out points that have fallen out of the src (moving) image's FOV
    msk = mask_grid(deformed_grid, src.shape)
    deformed_grid = deformed_grid[..., msk]
    _ref = ref[msk]

    # resample src image on deformed grid, thereby warping the former
    warped_src = trilinear_interp(src.ravel(order='F'), src.shape,
                                  deformed_grid[0], deformed_grid[1],
                                  deformed_grid[2])

    # compute joint histogram proper
    # XXX all the bottle neck is in this call to numpy's histogram2d
    return np.histogram2d(_ref, warped_src, bins=bins)[0]
