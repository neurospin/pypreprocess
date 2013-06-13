"""
:Module: affine_transformations
:Synopsis: routine functions for doing affine
transformation-related business
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
import scipy.linalg

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

    # affine transformation
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


def coords(p, M1, M2, x1, x2, x3):
    """Rigidly transforms the current set of coordinates (working grid)
    according to current motion estimates, p

    """

    # build coordinate transformation (matrix for passing from M2 space to
    # M1 space)
    M = np.dot(scipy.linalg.inv(np.dot(spm_matrix(p), M2)), M1)

    # apply the transformation
    _x1 = M[0, 0] * x1 + M[0, 1] * x2 + M[0, 2] * x3 + M[0, 3]
    _x2 = M[1, 0] * x1 + M[1, 1] * x2 + M[1, 2] * x3 + M[1, 3]
    _x3 = M[2, 0] * x1 + M[2, 1] * x2 + M[2, 2] * x3 + M[2, 3]

    return np.array([_x1, _x2, _x3])


def get_physical_coords(affine, point):
    point = np.array(point).reshape((-1, 3))

    return coords([0, 0, 0, 0, 0, 0], affine, np.eye(4), point[..., 0],
                  point[..., 1], point[..., 2]).reshape(point.shape).T
