"""
:Module: affine_transformations
:Synopsis: routine functions for doing affine transformation-related business
:Author: DOHMATOB Elvis Dopgima

References
----------
[1] Rigid Body Registration, by J. Ashburner and K. Friston

"""

import numpy as np
import scipy.linalg
import nibabel
from .io_utils import (load_specific_vol,
                       load_vol,
                       load_vols
                       )

# house-hold convenience naming
MOTION_PARAMS_NAMES = {0x0: 'Tx',
                       0x1: 'Ty',
                       0x2: 'Tz',
                       0x3: 'Rx',
                       0x4: 'Ry',
                       0x5: 'Rz',
                       0x6: 'Zx',
                       0x7: 'Zy',
                       0x8: 'Zz',
                       0x9: 'Sx',
                       0xA: 'Sy',
                       0xB: 'Sz',
                       }


def get_initial_motion_params():
    """Returns an array of length 12 (3 translations, 3 rotations, 3 zooms,
    and 3 shears) corresponding to the identity orthogonal transformation
    eye(4). Thus this is precisely made of: no zeto translation, zero rotation,
    a unit zoom of in each coordinate direction, and zero shear.

    """

    p = np.zeros(12)  # zero everything
    p[6:9] = 1.  # but unit zoom

    return p


def spm_matrix(p, order='T*R*Z*S'):
    """spm_matrix returns a matrix defining an orthogonal linear (translation,
    rotation, scaling and/or shear) transformation given a vector of
    parameters (p).  By default, the transformations are applied in the
    following order (i.e., the opposite to which they are specified):

    1) shear
    2) scale (zoom)
    3) rotation - yaw, roll & pitch
    4) translation

    This order can be changed by calling spm_matrix with a string as a
    second argument. This string may contain any valid MATLAB expression
    that returns a 4x4 matrix after evaluation. The special characters 'S',
    'Z', 'R', 'T' can be used to reference the transformations 1)-4)
    above.

    Parameters
    ----------
    p: array_like of length 12
        vector of parameters
        p(0)  - x translation
        p(1)  - y translation
        p(2)  - z translation
        p(3)  - x rotation about - {pitch} (radians)
        p(4)  - y rotation about - {roll}  (radians)
        p(5)  - z rotation about - {yaw}   (radians)
        p(6)  - x scaling
        p(7)  - y scaling
        p(8)  - z scaling
        p(9) - x shear
        p(10) - y shear
        p(11) - z shear
    order: string optional (default 'T*R*Z*S')
        application order of transformations.

    Returns
    -------
    A: 2D array of shape (4, 4)
        orthogonal transformation matrix

    """

    # fill-up up p to length 12, if too short
    q = get_initial_motion_params()
    p = np.hstack((p, q[len(p):12]))

    # translation
    T = np.eye(4)
    T[:3, -1] = p[:3]

    # yaw
    R1 = np.eye(4)
    R1[1, 1:3] = np.cos(p[3]), np.sin(p[3])
    R1[2, 1:3] = -np.sin(p[3]), np.cos(p[3])

    # roll
    R2 = np.eye(4)
    R2[0, [0, 2]] = np.cos(p[4]), np.sin(p[4])
    R2[2, [0, 2]] = -np.sin(p[4]), np.cos(p[4])

    # pitch
    R3 = np.eye(4)
    R3[0, 0:2] = np.cos(p[5]), np.sin(p[5])
    R3[1, 0:2] = -np.sin(p[5]), np.cos(p[5])

    # rotation
    R = np.dot(R1, np.dot(R2, R3))

    # zoom
    Z = np.eye(4)
    np.fill_diagonal(Z[0:3, 0:3], p[6:9])

    # scaling
    S = np.eye(4)
    S[0, 1:3] = p[9:11]
    S[1, 2] = p[11]

    # compute the complete affine transformation
    M = np.dot(T, np.dot(R, np.dot(Z, S)))

    return M


def spm_imatrix(M):
    """Returns parameters the 12 parameters for creating a given affine
    transformation.

    Parameters
    ----------
    M: array_like of shape (4, 4)
        affine transformation matrix

   Returns
   -------
   p: 1D array of length 12
       parameters for creating the afine transformation M

   """

    # there may be slight rounding errors making |b| > 1
    def rang(b):
        return min(max(b, -1), 1)

    # translations and zooms
    R = M[:3, :3]
    C = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)
    p = np.hstack((M[:3, 3], [0, 0, 0], np.diag(C), [0, 0, 0]))

    # handle case of -ve determinant
    if scipy.linalg.det(R) < 0:
        p[6] *= -1.

    # shears
    C = scipy.linalg.lstsq(np.diag(np.diag(C)), C)[0]
    p[9:12] = C.ravel()[[3, 6, 7]]
    R0 = spm_matrix(np.hstack(([0, 0, 0, 0, 0, 0], p[6:12])))
    R0 = R0[:3, :3]
    R1 = np.dot(R, scipy.linalg.inv(R0))

    # this just leaves rotations in matrix R1
    p[4] = np.arcsin(rang(R1[0, 2]))
    if (np.abs(p[4]) - np.pi / 2.) ** 2 < 1e-9:
        p[3] = 0.
        p[5] = np.arctan2(-rang(R1[1, 0]), rang(-R1[2, 0] / R[1, 3]))
    else:
        c = np.cos(p[4])
        p[3] = np.arctan2(rang(R1[1, 2] / c), rang(R1[2, 2] / c))
        p[5] = np.arctan2(rang(R1[0, 1] / c), rang(R1[0, 0] / c))

    # return parameters
    return p


def transform_coords(p, M1, M2, coords):
    """Rigidly transforms the current set of coordinates (working grid)
    according to current motion estimates, p, from one native space M1
    to another M2

    Parameters
    ----------
    p: 1D array_like of length at most 12
       current motion estimates
    M1: 2D array_like of shape (4, 4)
        affine definining the source space
    M2: 2D array_like of shape (4, 4)
        affine definining the destination space
    coords: 2D array_like of shape (3, n_voxels)
        the coordinates of the voxel(s) being transformed

    Returns
    -------
    array of same shape as the input coords:
        the transformed coordinates

    """

    # sanitize coords
    coords = np.reshape(coords, (3, -1))
    coords_shape = coords.shape

    # append row of ones
    coords = np.vstack((coords, np.ones(coords.shape[1])))

    # build coordinate transformation (matrix for passing from M2 space to
    # M1 space)
    M = np.dot(scipy.linalg.inv(np.dot(spm_matrix(p), M2)), M1)

    # apply the transformation
    return np.dot(M, coords)[:-1].reshape(coords_shape)


def get_physical_coords(M, voxel):
    """Get the scanner (world) coordinates of a voxel (or set of voxels) in
    the brain.

    Parameters
    ----------
    M: 2D array of shape (4, 4)
        affine transformation describing voxel-to-world mapping
    voxel: array_like of shape (3, n_voxels)
        voxel(s) under consideration

    Returns
    -------
    array of same shape as voxel input

    """

    # sanitize voxel dim
    voxel = np.array(voxel).reshape((3, -1))

    # compute and return the coords
    return transform_coords(np.zeros(6), M, np.eye(4), voxel)


def nibabel2spm_affine(affine):
    """
    SPM uses MATLAB indexing convention, so affine zero in 3D is [1, 1, 1, 1].
    Nibabel uses python indexing convention and so [0, 0, 0, 1] is affine zero.
    Thus to work in SPM algebra, we need to cast

        [0, 0, 0, 1] -> [-1, -1, -1, 1],

    that is, we need to dope the (nibabel) affine matrix accordingly. This
    function does just that.

    Notes
    -----
    This function should be used to prepare nifti affine matrices prior
    to python out python implementations of the SPM registration algorithms.
    This is done back-end, and the front-end user shouldn't see this.
    In fact, this function is low-level, and should be shealded from the user.

    """

    assert affine.shape == (4, 4)

    zero = [-1, -1, -1, 1]  # weird, huh ?
    affine[..., -1] = np.dot(affine, zero)

    return affine


def apply_realignment_to_vol(vol, q, inverse=True):
    """
    Modifies the affine headers of the given volume according to
    the realignment parameters (q).

    Parameters
    ----------
    vol: `nibabel.Nifti1Image`
        image to be transformed
    q: 1D array of length <= 12
        realignment parameters representing the rigid transformation
    inverse: boolean, optional (default False)
        indicates the direction in which the transformation is to be performed;
        if set then, it is assumed q actually represents the inverse of the
        transformation to be applied

    Returns
    -------
    `nibabel.Nifti1Image` object
        the realigned volume

    Notes
    -----
    Input is not modified.

    """

    vol = load_vol(vol)

    # convert realigment params to affine transformation
    M_q = spm_matrix(q)

    if inverse:
        M_q = scipy.linalg.inv(M_q)

    # apply affine transformation
    rvol = nibabel.Nifti1Image(vol.get_data(), np.dot(
            M_q, vol.get_affine()))

    rvol.get_data()

    return rvol


def apply_realignment(vols, rp, inverse=True):
    """
    Modifies  according to
    the realignment parameters (rp).

    vols: `nibabel.Nifti1Image`
        volumes to be transformed

    rp: 2D array of shape (n_vols, k) or (1, k), where k <=12
        realignment parameters representing the rigid transformations to be
        applied to the respective volumes.

    inverse: boolean, optional (default False)
        indicates the direction in which the transformation is to be performed;
        if set then, it is assumed q actually represents the inverse of the
        transformation to be applied

    Returns
    -------
    generator of `nibabel.Nifti1Image` objects
        the realigned volumes

    Notes
    -----
    Input is not modified.

    """

    # get number of scans
    _, n_scans = load_specific_vol(vols, 0)

    if n_scans == 1:
        vols = [load_specific_vol(vols, 0)[0]]

    # sanitize rp
    rp = np.array(rp)
    if rp.ndim == 1:
        rp = np.array([rp] * n_scans)

    rvols = [apply_realignment_to_vol(vol, rp[t], inverse=inverse)
             for vol, t in zip(load_vols(vols),
                               xrange(n_scans))]

    return rvols if n_scans > 1 or isinstance(vols, list) else rvols[0]


def extract_realignment_params(ref_vol, vol):
    """
    Extracts realignment param for vol -> ref_vol rigid body registration

    """

   # store estimated motion for volume t
    return spm_imatrix(
        np.dot(vol.get_affine(), scipy.linalg.inv(ref_vol.get_affine()))
               )
