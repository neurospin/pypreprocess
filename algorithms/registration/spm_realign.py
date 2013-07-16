"""
:Module: spm_realign
:Synopsis: MRI within-modality Motion Correction, SPM style
:Author: DOHMATOB Elvis Dopgima

References
----------
[1] Rigid Body Registration, by J. Ashburner and K. Friston

"""

import nibabel
import os
import numpy as np
import scipy.ndimage as ndi
import scipy.linalg
import kernel_smooth
import affine_transformations
import spm_reslice

# useful constants
INFINITY = np.inf
GRID_NOISE_AMPLITUDE = 1e-3  # amplitude of white-noise for doping the grid


def _load_vol(x):
    """
    Loads a single 3D volume.

    """

    if isinstance(x, basestring):
        vol = nibabel.load(x)
    elif isinstance(x, nibabel.Nifti1Image) or isinstance(
        x, nibabel.Nifti1Pair):
        vol = x
    else:
        raise TypeError(
            ("Each volume must be string, image object, got:"
             " %s") % type(x))

    if len(vol.shape) == 4:
        if vol.shape[-1] == 1:
            vol = nibabel.Nifti1Image(vol.get_data()[..., 0],
                                      vol.get_affine())
        else:
            raise ValueError(
                "Each volume must be 3D, got %iD" % len(vol.shape))
    elif len(vol.shape) != 3:
            raise ValueError(
                "Each volume must be 3D, got %iD" % len(vol.shape))

    return vol


def _load_specific_vol(vols, t):
    """Utility function for loading specific volume on demand.

    """

    assert t >= 0

    if isinstance(vols, list):
        n_scans = len(vols)
        vol = _load_vol(vols[t])
    elif isinstance(vols, nibabel.Nifti1Image) or isinstance(
        vols, nibabel.Nifti1Pair) or isinstance(vols, basestring):
        _vols = nibabel.load(vols) if isinstance(vols, basestring) else vols
        if len(_vols.shape) != 4:
            raise ValueError(
                "Expecting 4D image, got %iD" % len(_vols.shape))

        n_scans = _vols.shape[3]
        vol = nibabel.four_to_three(_vols)[t]
    else:  # unhandled type
        raise TypeError(
            "imgs arg must be string, image object, or list of such.")

    return vol, n_scans


def compute_rate_of_change_of_chisq(M, coords, gradG, lkp=xrange(6)):
    """Constructs matrix of rate of change of chi2 w.r.t. parameter changes

    Parameters
    -----------
    M: 2D array of shape (4, 4)
        affine matrix of source image
    coords: 2D array_like of shape (3, n_voxels)
        number of voxels in the problem (i.e on the working grid)
    gradG: 2D array_like of shape (3, n_voxels)
        gradient of reference volume, computed at the coords
    lkp: 1D array of length not greather than 12, optional (default
    [0, 1, 2, 3, 4, 5])
        motion parameters we're interested in

    Returns
    -------
    A: 2D array of shape (len(x1), len(lkp))

    """

    # # ravel gradient
    # gradG = np.reshape(gradG, (3, -1))

    # # ravel coords
    # coords = np.reshape(coords, (3, -1))

    # loop over parameters (this loop computes columns of A)
    A = np.ndarray((coords.shape[1], len(lkp)))
    for i in xrange(len(lkp)):
        pt = affine_transformations.get_initial_motion_params()

        # regularization ith parameter
        pt[lkp[i]] = pt[i] + 1e-6

        # map cartesian coordinate space according to motion
        # parameters pt (using jacobian associated with the transformation)
        transformed_coords  = affine_transformations.transform_coords(pt, M, M,
                                                                      coords)

        # compute change in cartesian coordinates as a result of change in
        # motion parameters
        dcoords = transformed_coords - coords

        # compute change in image intensities, as a result of the a change
        # in motion parameters
        dG = gradG * dcoords

        # compute ith column of A
        A[..., i] = dG.T.sum(axis=1)

        # regularization
        A[..., i] /= -1e-6

    return A


class MRIMotionCorrection(object):
    """Implements within-modality multi-session rigid-body registration of MRI
    volumes.

    The fit(...) method  estimates affine transformations necessary to
    rigidly align the other volumes to the first volume (hereafter referred
    to as the reference), whilst the transform(...) method actually writes
    these realigned files unto disk, optionally reslicing them.

    """

    def __init__(self, sep=4, interp=3, fwhm=5., quality=.9, tol=1e-8,
                 lkp=[0, 1, 2, 3, 4, 5], verbose=1,
                 n_iterations=64,
                 n_sessions=1
                 ):
        """Default constructor.

        quality: float, optional (default .9)
            quality versus speed trade-off.  Highest quality (1) gives most
            precise results, whereas lower qualities gives faster realignment.
            The idea is that some voxels contribute little to the estimation
            of the realignment parameters. This parameter is involved in
            selecting the number of voxels that are used.
        tol: float, optional (defaul 1e-8)
            tolerance for Gauss-Newton LS iterations
        fwhm: float, optional (default 10)
            the FWHM of the Gaussian smoothing kernel (mm) applied to the
            images before estimating the realignment parameters.
        sep: intx, optional (default 4)
            the default separation (mm) to sample the images
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
        verbose: int, optional (default 1)
            controls verbosity level. 0 means no verbose at all
        output_dir: string, optional (default None)
            existing dirname, where all output (i.e realigned files) will be
            written
        prefix: string, optional (default 'r')
            prefix to be prepended to output file basenames
        n_iterations: int, optional (dafault 64)
            max number of Gauss-Newton iterations when solving LSP for
            registering a volume to the reference

        """

        self._sep = sep
        self._interp = interp
        self._fwhm = fwhm
        self._quality = quality
        self._tol = tol
        self._lkp = lkp
        self._verbose = verbose
        self._n_iterations = n_iterations
        self._n_sessions = n_sessions

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

    def _single_session_fit(self, vols, quality=None,
                            affine_correction=np.eye(4)):
        """Realigns volumes (vols) from a single session.

        Parameters
        ----------
        vols: list of `nibabel.Nifti1Image` objects
            the volumes to realign
        quality: float, optional (default None)
            to override instance value

        Returns
        -------

        Notes
        -----
        Input vols is modified!

        """
        if quality is None:
            quality = self._quality

        # load first vol
        vol_0, n_scans = _load_specific_vol(vols, 0)

        # single vol ?
        if n_scans < 2:
            return np.array(
                [affine_transformations.get_initial_motion_params()])

        # affine correction
        vol_0 = nibabel.Nifti1Image(vol_0.get_data(),
                                    np.dot(affine_correction,
                                           vol_0.get_affine()))

        # voxel dimensions on the working grid
        skip = np.sqrt(np.sum(vol_0.get_affine()[:3, :3] ** 2, axis=0)
                       ) ** (-1) * self._sep

        # build working grid
        dim = vol_0.shape
        x1, x2, x3 = np.mgrid[0:dim[0] - .5 - 1:skip[0],
                              0:dim[1] - .5 - 1:skip[1],
                              0:dim[2] - .5 - 1:skip[2]].reshape((3, -1))

        # dope the grid so we're not perfectly aligned with its vertices
        # x1 += np.random.randn(*x1.shape) * GRID_NOISE_AMPLITUDE
        # x2 += np.random.randn(*x2.shape) * GRID_NOISE_AMPLITUDE
        # x3 += np.random.randn(*x3.shape) * GRID_NOISE_AMPLITUDE

        # smooth 0th volume to absorb noise before differentiating
        sref_vol = kernel_smooth.smooth_image(vol_0,
                                              self._fwhm).get_data()

        # resample the smoothed reference volume unto doped working grid
        G = ndi.map_coordinates(sref_vol, [x1, x2, x3], order=self._interp,
                                mode='wrap',).reshape(x1.shape)

        # compute gradient of reference volume
        Gx, Gy, Gz = np.gradient(sref_vol)

        # resample gradient unto working grid
        Gx = ndi.map_coordinates(Gx, [x1, x2, x3], order=self._interp,
                                 mode='wrap',).reshape(x1.shape)
        Gy = ndi.map_coordinates(Gy, [x1, x2, x3], order=self._interp,
                                 mode='wrap',).reshape(x1.shape)
        Gz = ndi.map_coordinates(Gz, [x1, x2, x3], order=self._interp,
                                 mode='wrap',).reshape(x1.shape)

        # compute rate of change of chi2 w.r.t. parameters
        A0 = compute_rate_of_change_of_chisq(vol_0.get_affine(),
                                             [x1, x2, x3], [Gx, Gy, Gz])

        # compute intercept vector for LSPs
        b = G.ravel()

        # Remove voxels that contribute very little to the final estimate.
        # It's essentially sufficient to remove voxels that contribute the
        # least to the determinant of the inverse convariance matrix
        # (i.e of the precision matrix)
        if n_scans > 2:
            self._log(
                ("Eliminating unimportant voxels (target quality: %s)"
                 "...") % quality)
            alpha = np.vstack((A0.T, b)).T
            alpha = np.dot(alpha.T, alpha)
            det0 = scipy.linalg.det(alpha)
            det1 = det0  # det1 / det0 is a precision measure
            n_eliminated_voxels = 0
            while det1 / det0 > quality:
                # determine unimportant voxels to eliminate
                dets = np.ndarray(A0.shape[0])
                for t in xrange(A0.shape[0]):
                    tmp = np.hstack((A0[t, ...], b[t])).reshape(
                        (1, A0.shape[1] + 1))
                    dets[t] = scipy.linalg.det(alpha - np.dot(tmp.T, tmp))
                msk = np.argsort(det1 - dets)
                msk = msk[:np.round(len(dets) / 10.)]

                # eliminate unimportant voxels
                n_eliminated_voxels += len(msk)
                self._log(
                    "\tEliminating %i voxels (current quality = %s)..." % (
                        len(msk), det1 / det0))
                A0 = np.delete(A0, msk, axis=0)
                b = np.delete(b, msk, axis=0)
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
            self._log(
                "...done; eliminated %i voxels.\r\n" % n_eliminated_voxels)

        # register the volumes to the reference volume
        self._log("Registering volumes to reference ( = volume 1)...")
        rp = np.ndarray((n_scans, 12))
        rp[0, ...] = affine_transformations.get_initial_motion_params(
            )  # don't mov the reference image
        ref_iaffine = scipy.linalg.inv(vol_0.get_affine())
        for t in xrange(1, n_scans):
            self._log("\tRegistering volume %i/%i..." % (
                    t + 1, n_scans))

            # load volume t
            vol, _ = _load_specific_vol(vols, t)
            vol = nibabel.Nifti1Image(vol.get_data(),
                                      np.dot(affine_correction,
                                             vol.get_affine()))

            # smooth volume t
            V = kernel_smooth.smooth_image(vol,
                                           self._fwhm).get_data()

            # intialize motion params for this vol
            rp[t, ...] = affine_transformations.get_initial_motion_params()

            # global optical flow problem with affine motion model: run
            # Gauss-Newton iterated LS (this loop should normally converge
            # after about as few as 5 iterations)
            dim = np.hstack((V.shape, [1, 1]))
            dim = dim[:3]
            ss = INFINITY
            countdown = -1
            for iteration in xrange(self._n_iterations):
                # starting point
                q = affine_transformations.get_initial_motion_params()

                # pass from volume t's grid to that of the reference
                # volume (0)
                y1, y2, y3 = affine_transformations.transform_coords(
                    np.zeros(6), vol_0.get_affine(),
                    vol.get_affine(), [x1, x2, x3])

                # sanity mask: some voxels might have fallen out of business;
                # and zap'em
                msk = np.nonzero((y1 >= 0) & (y1 < dim[0]) & (y2 >= 0)
                                 & (y2 < dim[1]) & (y3 >= 0)
                                 & (y3 < dim[2]))[0]

                # if mask is too small, then we're screwed anyway
                if len(msk) < 32:
                    raise RuntimeError(
                        ("Almost all voxels have fallen out of the FOV. Only "
                         "%i voxels survived. Registration can't work." % len(
                                msk)))

                # warp: resample volume t on this new grid
                F = ndi.map_coordinates(V, [y1[msk], y2[msk], y3[msk]],
                                        order=3, mode='wrap')

                # formulate and solve LS problem for updating p
                A = A0[msk, ...].copy()
                b1 = b[msk].copy()
                sc = np.sum(b1) / np.sum(F)
                b1 -= F * sc
                q_update = scipy.linalg.lstsq(np.dot(A.T, A),
                                              np.dot(A.T, b1))[0]

                # update q
                q[self._lkp] += q_update

                # get affine transformation represented by q
                M_q = affine_transformations.spm_matrix(q)

                # update affine matrix for volume t by applying M_q
                vol = nibabel.Nifti1Image(
                    vol.get_data(), np.dot(scipy.linalg.inv(M_q),
                                             vol.get_affine()))

                # compute convergence criterion variables
                pss = ss
                ss = np.sum(b1 ** 2) / len(b1)

                # compute relative gain over last iteration
                relative_gain = np.abs((pss - ss) / pss
                                       ) if np.isfinite(pss) else INFINITY

                # update progress bar
                token = "\t\t" + "   ".join(['%-8.4g' % x
                                            for x in q[self._lkp]])
                token += " " * (len(self._lkp) * 13 - len(token)
                                ) + "| %.5g" % relative_gain
                self._log(token)

                # check whether we've stopped converging altogether
                if relative_gain < self._tol and countdown == -1:
                    countdown = 2

                # countdown
                if countdown != -1:
                    # converge
                    if countdown == 0:
                        break
                    countdown -= 1

            # what happened ?
            comments = " after %i iterations" % (iteration + 1)
            if iteration + 1 == self._n_iterations:
                comments = "did not coverge" + comments
            else:
                if relative_gain < self._tol:
                    comments = "converged" + comments
                else:
                    comments = "stopped converging" + comments

            # store estimated motion for volume t
            rp[t, ...] = affine_transformations.spm_imatrix(
                np.dot(vol.get_affine(), ref_iaffine)
                # scipy.linalg.lstsq(vol.get_affine(), vol_0.get_affine())[0]
                )
            self._log("\t\t...done; %s.\r\n" % comments)

        return rp

    def fit(self, vols):
        """Estimation of within-modality rigid-body movement parameters.

        All operations are performed relative to the first image. That is,
        registration is to the first image, and resampling of images is
        into the space of the first image.

        The algorithm is a Gauss-Netwon iterative refinememnt of an l2 loss
        (squared difference between images)

        Parameters
        ----------
        vols: list of lists
            list of single 4D images or nibabel image objects
            (one per session), or list of lists of filenames or nibabel image
            objects (one list per session)

        Raises
        ------
        RuntimeError, ValueError

        """

        # sanitize vols and n_sessions
        if isinstance(vols, basestring) or isinstance(
            vols, nibabel.Nifti1Image):
            vols = [vols]
        if len(vols) != self._n_sessions:
            raise RuntimeError(
                "Number of session volumes (%i) != number of sessions (%i)"
                % (len(vols), self._n_sessions))

        self._vols = vols

        # load first vol of each session
        first_vols = []
        n_scans_list = []
        for sess in xrange(self._n_sessions):
            vol_0, n_scans = _load_specific_vol(self._vols[sess], 0)
            first_vols.append(vol_0)
            n_scans_list.append(n_scans)
        self._n_scans_list = n_scans_list

        # realign first vol of each session with first vol of first session
        if self._n_sessions > 1:
            self._log(
                ('\r\nInter-session registration: realigning first volumes'
                 ' of all sessions...'))

        self._first_vols_rp = self._single_session_fit(
            first_vols,
            quality=1.  # only a few vols, we can thus allow this lux
            )

        rfirst_vols = []
        for sess in xrange(self._n_sessions):
            vol = first_vols[sess]

            if sess > 0:
                # grap realignment params for this vol
                q = self._first_vols_rp[sess]

                # convert realigment params to affine transformation
                M_q = affine_transformations.spm_matrix(q)

                # apply affine transformation
                rvol = nibabel.Nifti1Image(vol.get_data(), np.dot(
                        M_q, vol.get_affine()))
            else:
                rvol = vol

            rfirst_vols.append(rvol)

        if self._n_sessions > 1:
            self._log('...done (inter-session registration).\r\n')

        # realign all vols of each session with first vol of that session
        self._rp = []
        for sess in xrange(self._n_sessions):
            self._log(
                ("Intra-session registration: Realigning session"
                 " %i/%i...") % (sess + 1, self._n_sessions))

            # affine correction, for inter-session realignment
            affine_correction = np.dot(rfirst_vols[sess].get_affine(),
                                  scipy.linalg.inv(
                    rfirst_vols[0].get_affine()))

            sess_rp = self._single_session_fit(
                self._vols[sess],
                affine_correction=affine_correction)

            self._rp.append(sess_rp)
            self._log('...done; session %i.\r\n' % (sess + 1))

    def transform(self, output_dir, reslice=False, prefix="r",
                  ext=".nii.gz", concat=False):
        """Saves realigned volumes and the realigment parameters to disk.
        Realigment parameters are stored in output_dir/rp.txt and Volumes
        are saved in output_dir/prefix_vol.ext where and t is the scan
        number of the corresponding (3D) volume.

        Parameters
        ----------
        reslice: bool, optional (default False)
            reslice the realigned volumes
        output_dir: string, optional (dafault None)
            existing dirname where output will be written
        prefix: string, optional (default 'r')
            prefix for output filenames.
        ext: string, optional (default ".nii.gz")
            file extension for ouput images
        concat: boolean, optional (default False)
            concatenate the ouput volumes for each session into a single
            4D film

        Returns
        -------
        output: dict
            output dict. items are:
            rvols: list of `nibabel.Nifti1Image` objects
                list of realigned 3D vols

            And if output_dir is not None, output will also have the
            following items:
            realigned_files: list of strings
                full paths of the realigned files
            rp_filename: string
                full path of text file containing realignment parameters

        """

        def _save_session_realigned_vols(sess_realigned_vols,
                                        sess_output_dir):
            if concat:
                output_filename = os.path.join(
                    sess_output_dir, "%s_4D%s" % (prefix, ext))
                nibabel.save(nibabel.concat_images(sess_realigned_vols),
                             output_filename)
                sess_realigned_files = output_filename
            else:
                sess_realigned_files = []
                for t in xrange(self._n_scans_list[sess]):
                    # save realigned vol unto disk
                    output_filename = os.path.join(sess_output_dir,
                                                   "%s_vol_%i%s" % (
                            prefix, t, ext))
                    nibabel.save(sess_realigned_vols[t], output_filename)

                    # update rvols and realigned_files
                    sess_realigned_files.append(output_filename)

            return sess_realigned_files

        output = {'rp': self._rp}

        # sanitize ext
        if not ext.startswith('.'):
            ext = "." + ext

        output['realigned_files'] = []
        output['rp_filenames'] = []
        sess_output_dirs = []

        for sess in xrange(self._n_sessions):
            sess_realigned_files = []
            sess_output_dir = os.path.join(output_dir, "%i" % sess)
            sess_output_dirs.append(sess_output_dir)

            # make session output dir
            if not os.path.exists(sess_output_dir):
                os.makedirs(sess_output_dir)

            # modify the header of each 3D vols according to the
            # estimated motion (realignment params)
            sess_rvols = []
            for t in xrange(self._n_scans_list[sess]):
                vol, _ = _load_specific_vol(self._vols[sess], t)

                # grap realignment params for this vol
                q = self._rp[sess][t]

                # convert realigment params to affine transformation
                M_q = affine_transformations.spm_matrix(q)

                # apply affine transformation
                vol = nibabel.Nifti1Image(vol.get_data(), np.dot(
                        M_q, vol.get_affine()))

                sess_rvols.append(vol)

            # reslice vols
            if reslice:
                self._log('Reslicing volumes for session %i/%i...' % (
                        sess + 1, self._n_sessions))
                sess_rvols = list(spm_reslice.reslice_vols(sess_rvols))
                self._log('...done; session %i/%i.' % (
                        sess + 1, self._n_sessions))

            # save realigned files to disk
            sess_realigned_files = _save_session_realigned_vols(
                sess_rvols,
                sess_output_dir)
            output['realigned_files'].append(sess_realigned_files)

            # save realignment params to disk
            sess_rp_filename = os.path.join(sess_output_dir, "rp.txt")
            np.savetxt(sess_rp_filename, self._rp[sess][..., self._lkp])
            output['rp_filenames'].append(sess_rp_filename)

        self._log('...done; output saved to %s.' % output_dir)

        # return output
        return output
