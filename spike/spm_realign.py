"""
:Module: spm_realign
:Synopsis: motion correction SPM style
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.interpolate
import scipy.linalg
from nipy.algorithms.kernel_smooth import LinearFilter
import nipy
import matplotlib.pyplot as plt
import splines_at_war as saw
import glob

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


def coords(p, M1, M2, xx, yy, zz):
    """Rigidly transforms the current set of coordinates (working grid)
    according to current motion estimates, p

    """

    # build coordinate transformation (matrix for passing from M2 space to
    # M1 space)
    M = np.dot(scipy.linalg.inv(np.dot(spm_matrix(p), M2)), M1)

    # apply the transformation
    _xx = M[0, 0] * xx + M[0, 1] * yy + M[0, 2] * zz + M[0, 3]
    _yy = M[1, 0] * xx + M[1, 1] * yy + M[1, 2] * zz + M[1, 3]
    _zz = M[2, 0] * xx + M[2, 1] * yy + M[2, 2] * zz + M[2, 3]

    return _xx, _yy, _zz


def make_A(M, xx, yy, zz, Gx, Gy, Gz, lkp=xrange(6), condensed_coords=False):
    """Constructs matrix of rate of change of chi2 w.r.t. parameter changes

    Parameters
    -----------
    M: 2D array of shape (4, 4)
        affine matrix of source image
    xx: 1D array
        x coordinates of working grid
    yy: 1D array or same size as xx
        y coordinates of working grid
    zz: 1D array of same size ax xx
        z coordinates of working grid
    Gx: 1D array of of same size as xx
        gradient of volume at the points (xx[i], yy[i], zz[i]), along
        the x axis
    Gy: 1D array of of same size as xx
        gradient of volume at the points (xx[i], yy[i], zz[i]), along
        the y axis
    Gz: 1D array of of same size as xx
        gradient of volume at the points (xx[i], yy[i], zz[i]), along
        the z axis
    lkp: 1D array of length not greather than 12, optional (default
    [0, 1, 2, 3, 4, 5])
        motion parameters we're interested in

    Returns
    -------
    A: 2D array of shape (len(xx), len(lkp))

    """

    # sanity
    Gx = Gx.ravel()
    Gy = Gy.ravel()
    Gz = Gz.ravel()

    if condensed_coords:
        grid = np.array([(x, y, z) for x in xx for y in yy for z in zz])
        xx = grid[..., 0]
        yy = grid[..., 1]
        zz = grid[..., 2]
    else:
        xx = xx.ravel()
        yy = yy.ravel()
        zz = zz.ravel()

    A = np.ndarray((len(xx), len(lkp)))

    # loop over parameters (this loop computes columns of A)
    for i in xrange(len(lkp)):
        pt = get_initial_motion_params()

        # regularization ith parameter
        pt[lkp[i]] = pt[i] + 1e-6

        # map cartesian coordinate space according to motion
        # parameters pt (using jacobian associated with the transformation)
        x, y, z = coords(pt, M, M, xx, yy, zz)

        # compute change in cartesian coordinates as a result of change in
        # motion parameters
        dspace = np.vstack((x - xx, y - yy, z - zz))

        # compute change in image intensities, as a result of the a change
        # in motion parameters
        dG = np.vstack((Gx, Gy, Gz)) * dspace

        # compute ith column of A
        A[..., i] = dG.T.sum(axis=1)

        # regularization
        A[..., i] /= -1e-6

    return A


def fit(fmri_filenames, sep=4, interp=2, fwhm=5., quality=.9, rtm=1,
           lkp=[1, 2, 3, 4, 5, 6]):
    """Don't do this @home!

    """

    # load data
    if isinstance(fmri_filenames, basestring):
        fmri_filenames = [fmri_filenames]
    nin = nipy.load_image(fmri_filenames[0])

    d = nin.shape[:3]
    skip = [1., 1., 1.]

    # compute coordinat map
    # XXX: xx, yyy and zz are not same as in SPM!
    xx, yy, zz = np.mgrid[:d[0]:skip[0], :d[1]:skip[1],
                           :d[2]:skip[2]]
    xx += np.random.randn(*xx.shape) * .5
    yy += np.random.randn(*yy.shape) * .5
    zz += np.random.randn(*zz.shape) * .5

    # compute rate of change of chi2 w.r.t. changes in parameters (matrix A)
    f = LinearFilter(nin.coordmap, nin.shape, fwhm=fwhm)
    svol = f.smooth(nin, clean=True)

    # resample the image and its gradient
    G = scipy.ndimage.map_coordinates(svol,
                                      [xx.ravel(), yy.ravel(), zz.ravel()],
                                      order=3,
                                      mode='wrap',).reshape(xx.shape)
    Gx = saw.compute_gradient_along_axis(svol, 0, grid=[xx, yy, zz],)
    Gy = saw.compute_gradient_along_axis(svol, 1, grid=[xx, yy, zz],)
    Gz = saw.compute_gradient_along_axis(svol, 2, grid=[xx, yy, zz],)

    # compute rate of change of chi2 w.r.t. parameters
    A0 = make_A(nin.coordmap.affine, xx, yy, zz, Gx, Gy, Gz,
                )

    return G, A0


if __name__ == '__main__':
    data_wildcat = "/home/elvis/CODE/datasets/henry_mondor/f38*.img"
    for vol_filename in sorted(glob.glob(data_wildcat)):
        print "Processing volume %s..." % vol_filename
        _, A0 = fit(vol_filename)

        # plot A0
        plt.figure()
        plt.plot(A0)
        plt.legend(tuple(MOTION_PARAMS_NAMES.values()))
        plt.xlabel("voxel")
        plt.ylabel('rate of change of Chi-squared w.r.t. motions parameters')
        plt.show()
        
