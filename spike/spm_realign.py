"""
:Module: spm_realign
:Synopsis: motion correction SPM style
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
import scipy.ndimage as ndi
import scipy.signal
import scipy.interpolate
import scipy.linalg
from nipy.algorithms.kernel_smooth import LinearFilter
import nipy
import matplotlib.pyplot as plt
import splines_at_war as saw
import glob
import os

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

    return _x1, _x2, _x3


def make_A(M, x1, x2, x3, Gx, Gy, Gz, lkp=xrange(6), condensed_coords=False):
    """Constructs matrix of rate of change of chi2 w.r.t. parameter changes

    Parameters
    -----------
    M: 2D array of shape (4, 4)
        affine matrix of source image
    x1: 1D array
        x coordinates of working grid
    x2: 1D array or same size as x1
        y coordinates of working grid
    x3: 1D array of same size ax x1
        z coordinates of working grid
    Gx: 1D array of of same size as x1
        gradient of volume at the points (x1[i], x2[i], x3[i]), along
        the x axis
    Gy: 1D array of of same size as x1
        gradient of volume at the points (x1[i], x2[i], x3[i]), along
        the y axis
    Gz: 1D array of of same size as x1
        gradient of volume at the points (x1[i], x2[i], x3[i]), along
        the z axis
    lkp: 1D array of length not greather than 12, optional (default
    [0, 1, 2, 3, 4, 5])
        motion parameters we're interested in

    Returns
    -------
    A: 2D array of shape (len(x1), len(lkp))

    """

    # sanity
    Gx = Gx.ravel()
    Gy = Gy.ravel()
    Gz = Gz.ravel()

    if condensed_coords:
        grid = np.array([(x, y, z) for x in x1 for y in x2 for z in x3])
        x1 = grid[..., 0]
        x2 = grid[..., 1]
        x3 = grid[..., 2]
    else:
        x1 = x1.ravel()
        x2 = x2.ravel()
        x3 = x3.ravel()

    A = np.ndarray((len(x1), len(lkp)))

    # loop over parameters (this loop computes columns of A)
    for i in xrange(len(lkp)):
        pt = get_initial_motion_params()

        # regularization ith parameter
        pt[lkp[i]] = pt[i] + 1e-6

        # map cartesian coordinate space according to motion
        # parameters pt (using jacobian associated with the transformation)
        y1, y2, y3  = coords(pt, M, M, x1, x2, x3)

        # compute change in cartesian coordinates as a result of change in
        # motion parameters
        dspace = np.vstack((y1 - x1, y2 - x2, y3 - x3))

        # compute change in image intensities, as a result of the a change
        # in motion parameters
        dG = np.vstack((Gx, Gy, Gz)) * dspace

        # compute ith column of A
        A[..., i] = dG.T.sum(axis=1)

        # regularization
        A[..., i] /= -1e-6

    return A


class Realign(object):
    """Implements within-modality rigd-body registration.

    """

    def __init__(self, sep=4, interp=3, fwhm=5.,
                 quality=.9, rtm=False, lkp=[0, 1, 2, 3, 4, 5],
                 PW=None,
                 verbose=1,
                 ):
        """Default constructor.

        quality: float, optional (default .9)
            quality versus speed trade-off.  Highest quality (1) gives most
            precise results, whereas lower qualities gives faster realignment.
            The idea is that some voxels contribute little to the estimation
            of the realignment parameters. This parameter is involved in
            selecting the number of voxels that are used.
        fwhm: float, optional (default 10)
            the FWHM of the Gaussian smoothing kernel (mm) applied to the
            images before estimating the realignment parameters.
        sep: int or list of ints, optional (default 4)
            the default separation (mm) to sample the images
        rtm: boolean, optional (default False)
            register to mean.  If field exists then a two pass procedure is
            to be used in order to register the images to the mean of the image
            after the first realignment
        interp: int, optional (default 3)
            B-spline degree used for interpolation
        lkp: arry_like of ints, optional (default [0, 1, 2, 3, 4, 5])
            affine transformation parameters sought-for. Possible values of
            elements of the list are:
            0  - x translation
            1  - y translation
            2  - z translation
            3  - x rotation about - {pitch} (radians)
            4  - y rotation about - {roll}  (radians)
            5  - z rotation about - {yaw}   (radians)
            6  - x scaling
            7  - y scaling
            8  - z scaling
            9  - x shear
            10 - y shear
            11 - z shear
        PW: string, optional (default None)
            a filename of a weighting image (reciprocal of standard deviation).
            If field does not exist, then no weighting is done.
        verbose: int, optional (default 1)
            controls verbosity level. 0 means no verbose at all

        """

        self._sep = sep
        self._interp = interp
        self._fwhm = fwhm
        self._quality = quality
        self._rtm = rtm
        self._lkp = lkp
        self._PW = PW
        self._verbose = verbose

        pass

    def _log(self, msg):
        """Logs a message, according to verbose level.

        Parameters
        ----------
        msg: string
            message to log

        """

        if self._verbose:
            print(msg)

    def fit(self, filenames):
        """Estimation of within modality rigid body movement parameters.
        All operations are performed relative to the first image. That is,
        registration is to the first image, and resampling of images is
        into the space of the first image.

        The algorithm is a Gauss-Netwon iterative refinememnt of an l2 loss
        (squared difference between images)

        Parameters
        ----------
        filenames: array_like of strings
            list of filenames (3D volumes) to be registered.

        Returns
        -------
        rp: 2D array of shape (n_scans, 6)
            realignment parameters

        Raises
        ------
        RuntimeError

        """

        # load data
        if isinstance(filenames, basestring):
            filenames = [filenames]
        P = [nipy.load_image(fmri_filename) for fmri_filename in filenames]

        skip = np.sqrt(np.sum(P[0].coordmap.affine[:3, :3] ** 2, axis=0)
                       ) ** (-1) * self._sep
        dim = P[0].shape[:3]

        # compute working grid
        x1, x2, x3 = np.mgrid[:dim[0] - .5 - 1:skip[0],
                               :dim[1] - .5 - 1:skip[1],
                               :dim[2] - .5 - 1:skip[2]]

        # dope the grid so we're not perfectly aligned with its vertices
        x1 += np.random.randn(*x1.shape) * .5
        x2 += np.random.randn(*x2.shape) * .5
        x3 += np.random.randn(*x3.shape) * .5

        # ravel working grid
        x1 = x1.ravel(order='C')
        x2 = x2.ravel(order='C')
        x3 = x3.ravel(order='C')

        # smooth volume to absorb noise before differentiating
        smoothing_kernel = LinearFilter(P[0].coordmap, P[0].shape,
                                        fwhm=self._fwhm)
        svol = smoothing_kernel.smooth(P[0], clean=True).get_data()

        # resample the image and its gradient
        G = ndi.map_coordinates(svol, [x1.ravel(), x2.ravel(), x3.ravel()],
                                order=self._interp,
                                mode='wrap',).reshape(x1.shape)

        # compute gradient of reference image
        Gx, Gy, Gz = np.gradient(svol)

        # resample gradients to working grid
        Gx = ndi.map_coordinates(Gx, [x1.ravel(), x2.ravel(), x3.ravel()],
                                order=self._interp,
                                mode='wrap',).reshape(x1.shape)
        Gy = ndi.map_coordinates(Gy, [x1.ravel(), x2.ravel(), x3.ravel()],
                                order=self._interp,
                                mode='wrap',).reshape(x1.shape)
        Gz = ndi.map_coordinates(Gz, [x1.ravel(), x2.ravel(), x3.ravel()],
                                order=self._interp,
                                mode='wrap',).reshape(x1.shape)

        # import scipy.io
        # X = scipy.io.loadmat('/media/Lexar/CODE/datasets/grad.mat',
        #                      squeeze_me=True,
        #                      struct_as_record=False)
        # Gx, Gy, Gz = [X[k] for k in ['dG1', 'dG2', 'dG3']]

        # # compute rate of change of chi2 w.r.t. parameters
        A0 = make_A(P[0].coordmap.affine, x1, x2, x3, Gx, Gy, Gz,
                )

        G = G.ravel(order='C')
        Gx = Gx.ravel(order='C')
        Gy = Gy.ravel(order='C')
        Gz = Gz.ravel(order='C')

        b = G.copy()

        # Remove voxels that contribute very little to the final estimate.
        # It's essentially sufficient to remove voxels that contribute the
        # least to the determinant of the inverse convariance matrix
        # (the precision matrix)
        self._log(
            ("Eliminating unimportant voxels (target quality: %s)"
             "...") % self._quality)
        alpha = np.vstack((A0.T, b)).T
        alpha = np.dot(alpha.T, alpha)
        det0 = scipy.linalg.det(alpha)
        det1 = det0  # det1 / det0 is a precision measure
        iteration = 0
        # XXX the following loop is unusually slow!!!
        while det1 / det0 > self._quality:
            iteration += 1
            self._log(
                "\tIteration %i: current quality = %s" % (iteration,
                                                          det1 / det0))

            # determine voxels unimportant voxels to eliminate
            dets = np.ndarray(A0.shape[0])
            for i in xrange(A0.shape[0]):
                tmp = np.hstack((A0[i, ...], b[i])).reshape((1,
                                                             A0.shape[1] + 1))
                dets[i] = scipy.linalg.det(alpha - np.dot(tmp.T, tmp))
            msk = np.argsort(det1 - dets)
            msk = msk[:np.round(len(dets) / 10.)]

            # eliminate unimportant voxels
            self._log(
                "\t\tEliminating %i voxels..." % len(msk))
            A0 = np.delete(A0, msk, axis=0)
            b = np.delete(b, msk, axis=0)
            G = np.delete(G, msk, axis=0)
            x1 = np.delete(x1, msk, axis=0)
            x2 = np.delete(x2, msk, axis=0)
            x3 = np.delete(x3, msk, axis=0)
            Gx = np.delete(Gx, msk, axis=0)
            Gy = np.delete(Gy, msk, axis=0)
            Gz = np.delete(Gz, msk, axis=0)

            # updates for next iteration
            alpha = np.vstack((A0.T, b)).T
            alpha = np.dot(alpha.T, alpha)
            det1 = scipy.linalg.det(alpha)

        self._log("Registering images to reference...")
        for i in xrange(1, len(P)):
            self._log("\tRegistering image %i/%i..." % (i, len(P) - 1))

            # smooth image i
            smoothing_kernel = LinearFilter(P[i].coordmap, P[i].shape,
                                            fwhm=self._fwhm)
            V = smoothing_kernel.smooth(P[i]).get_data()

            # Run Gauss-Newton Iterated LS (this should normally converge after
            # as few as 5 iterations
            dim = np.hstack((V.shape, [1, 1]))
            dim = dim[:3]
            ss = -np.inf
            countdown = -1
            for iteration in xrange(64):
                # pass from image i's grid to that of the reference image (0)
                y1, y2, y3 = coords([0, 0, 0, 0, 0, 0], P[0].coordmap.affine,
                                    P[i].coordmap.affine, x1, x2, x3)

                # sanity mask: some points might have fallen into hell, find'em
                # and zap'em
                msk = np.nonzero((y1 >= 0) & (y1 < dim[0]) & (y2 >= 0)
                                 & (y2 < dim[1]) & (y3 >= 0)
                                 & (y3 < dim[2]))[0]

                # if mask is too small, then we're screwed anyway
                if len(msk) < 32:
                    raise RuntimeError(
                        ("Mask too small, we can't learn anythx useful. "
                         "Game over!")
                        )

                # warp: resample image i on this new grid
                F = ndi.map_coordinates(V, [y1[msk], y2[msk], y3[msk]],
                                        order=3,
                                        mode='wrap',)

                # formulate and solve LS problem
                A = A0[msk, :].copy()
                b1 = b[msk].copy()
                sc = np.sum(b1) / np.sum(F)
                b1 -= F * sc
                soln = scipy.linalg.lstsq(np.dot(A.T, A), np.dot(A.T, b1))[0]

                # compute motion parameter estimates
                p = get_initial_motion_params()
                p[self._lkp] += soln.T

                # modify image i's by appling the learnt affine transformation
                P[i].coordmap.affine = np.dot(scipy.linalg.inv(spm_matrix(p)),
                                              P[i].coordmap.affine)

                # compute convergence criterion variables
                pss = ss
                ss = np.sum(b1 ** 2) / len(b1)

                # check whether we've stopped converging altogether
                relative_gain = np.abs((pss - ss) / pss)
                if relative_gain < 1e-8 and countdown == -1:
                    countdown = 2

                self._log("\t\trelative gain: %s" % relative_gain)

                # countdown
                if countdown != -1:
                    # converged ?
                    if countdown == 0:
                        break
                    countdown -= 1

        # extract the estimated motion params from the final affines
        rp = np.ndarray((len(P), 12))
        iref_affine = scipy.linalg.inv(P[0].coordmap.affine)
        for i in xrange(len(P)):
            rp[i, ...] = spm_imatrix(np.dot(P[i].coordmap.affine,
                                            iref_affine))

        # return estimated motion (realignment parameters)
        return rp


# demo for motion estimation
if __name__ == '__main__':
    data_wildcat = ("/home/elvis/CODE/datasets/spm_multimodal/fMRI/Session1"
                    "/fMETHODS-0005-0*.img")

    r = Realign()
    motion_parameters = r.fit(sorted(glob.glob(data_wildcat)))
