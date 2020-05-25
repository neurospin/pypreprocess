"""
:Module: spm_realign
:Synopsis: MRI within-modality Motion Correction, SPM style
:Author: DOHMATOB Elvis Dopgima

"""

import os
import numpy as np
import scipy.ndimage as ndimage
import scipy.linalg
import nibabel
from joblib import Parallel, delayed
from .kernel_smooth import smooth_image
from .affine_transformations import (
    get_initial_motion_params, transform_coords, apply_realignment_to_vol,
    apply_realignment)
from .reslice import reslice_vols
from .io_utils import save_vol, save_vols, get_basenames, is_niimg, load_vols

# useful constants
INFINITY = np.inf


def _single_volume_fit(moving_vol, fixed_vol_affine, fixed_vol_A0, affine_correction, b, x1, x2,
                       x3, fwhm, n_iterations, interp, lkp, tol, smooth_func=smooth_image,
                       log=lambda x: None):
    """
    Realigns moving_vol to fixed_vol.

    Paremeters
    ----------

    moving_vol: Nibabel object
        coregistration source

    fixed_vol_affine: 2D array
        coregistration target

    fixed_vol_A0: 2D array
        rate of change of chi2 w.r.t. parameter changes

    affine_correction: 2D array of shape (4, 4), optional (default None)
        affine transformation to be applied to vols before realignment
        (this is useful in multi-session realignment)

    b: array
        intercept vector

    x1, x2, x3: arrays
        grid

    fwhm: float
        the FWHM of the Gaussian smoothing kernel (mm) applied to the
        images before estimating the realignment parameters.

    n_iterations: int
        max number of Gauss-Newton iterations when solving LSP for
        registering a volume to the reference

    interp: int
        B-spline degree used for interpolation

    lkp: arry_like of ints
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

    tol: float
        tolerance for Gauss-Newton LS iterations

    smooth_func: function, optional (default pypreprocess' smooth_image)
        the smoothing function to apply during estimation. The given function
        must accept 2 positional args (vol, fwhm)

    log: function, optional (default lambda x: None)
        function used for storing log messages

    Returns
    -------
        1D array of length len(self.lkp)
            the estimated realignment parameters

    """
    moving_vol = nibabel.Nifti1Image(moving_vol.get_data(),
                                     np.dot(affine_correction,
                                            moving_vol.get_affine()))
    # initialize final rp for this vol
    vol_rp = get_initial_motion_params()
    # smooth volume t
    V = smooth_func(moving_vol, fwhm).get_data()
    # global optical flow problem with affine motion model: run
    # Gauss-Newton iterated LS (this loop should normally converge
    # after about as few as 5 iterations)
    dim = np.hstack((V.shape, [1, 1]))
    dim = dim[:3]
    ss = INFINITY
    countdown = -1
    iteration = None
    for iteration in range(n_iterations):
        # starting point
        q = get_initial_motion_params()

        # pass from volume t's grid to that of the reference
        # volume (0)
        y1, y2, y3 = transform_coords(np.zeros(6), fixed_vol_affine,
                                      moving_vol.get_affine(), [x1, x2, x3])

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
        F = ndimage.map_coordinates(V, [y1[msk], y2[msk], y3[msk]],
                                    order=interp, mode='wrap')

        # formulate and solve LS problem for updating p
        A = fixed_vol_A0[msk, ...].copy()
        b1 = b[msk].copy()
        sc = np.sum(b1) / np.sum(F)
        b1 -= F * sc
        q_update = scipy.linalg.lstsq(np.dot(A.T, A),
                                      np.dot(A.T, b1))[0]

        # update q
        q[lkp] += q_update
        vol_rp[lkp] -= q_update

        # update affine matrix for volume t by applying M_q
        moving_vol = apply_realignment_to_vol(moving_vol, q)

        # compute convergence criterion variables
        pss = ss
        ss = np.sum(b1 ** 2) / len(b1)

        # compute relative gain over last iteration
        relative_gain = np.abs((pss - ss) / pss
                               ) if np.isfinite(pss) else INFINITY

        # verbose
        token = "\t" + "".join(['%-12.4g ' % z for z in q[lkp]])
        token += '|  %.5g' % relative_gain
        log(token)

        # check whether we've stopped converging altogether
        if relative_gain < tol and countdown == -1:
            countdown = 2

        # countdown
        if countdown != -1:
            # converge
            if countdown == 0:
                break
            countdown -= 1

    # what happened ?
    comments = " after %i iterations" % (iteration + 1)
    if iteration + 1 == n_iterations:
        comments = "did not coverge" + comments
    else:
        if relative_gain < tol:
            comments = "converged" + comments
        else:
            comments = "stopped converging" + comments
    log(comments)
    return vol_rp


def _compute_rate_of_change_of_chisq(M, coords, gradG, lkp=range(6)):
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

    # sanitize input
    gradG = np.array(gradG)
    coords = np.array(coords)

    # loop over parameters (this loop computes columns of A)
    A = np.ndarray((coords.shape[1], len(lkp)))
    for i in range(len(lkp)):
        pt = get_initial_motion_params()

        # regularization ith parameter
        pt[lkp[i]] = pt[i] + 1e-6

        # map cartesian coordinate space according to motion
        # parameters pt (using jacobian associated with the transformation)
        transformed_coords = transform_coords(pt, M, M, coords)

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
    """
    Implements within-modality multi-session rigid-body registration of MRI
    volumes.

    The fit(...) method  estimates affine transformations necessary to
    rigidly align the other volumes to the first volume (hereafter referred
    to as the reference), whilst the transform(...) method actually writes
    these realigned files unto disk, optionally reslicing them.

    Paremeters
    ----------
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

    n_iterations: int, optional (dafault 64)
        max number of Gauss-Newton iterations when solving LSP for
        registering a volume to the reference

    smooth_func: function, optional (default pypreprocess' smooth_image)
        the smoothing function to apply during estimation. The given function
        must accept 2 positional args (vol, fwhm)

    Attributes
    ----------
    realignment_parameters_: 3D array of shape (n_sessions, n_scans_session, 6)
        the realigment parameters for each volume of each session

    References
    ----------
    [1] Rigid Body Registration, by J. Ashburner and K. Friston

    """

    def __init__(self, sep=4, interp=3, fwhm=5., quality=.9, tol=1e-8,
                 lkp=None, verbose=1, n_iterations=64, n_sessions=1,
                 smooth_func=smooth_image):
        lkp = [0, 1, 2, 3, 4, 5] if lkp is None else lkp
        self.sep = sep
        self.interp = interp
        self.fwhm = fwhm
        self.quality = quality
        self.tol = tol
        self.lkp = lkp
        self.verbose = verbose
        self.n_iterations = n_iterations
        self.n_sessions = n_sessions
        self.smooth_func = smooth_func

    def _log(self, msg):
        """Logs a message, according to verbose level.

        Parameters
        ----------
        msg: string
            message to log

        """

        if self.verbose:
            print(msg)

    def __repr__(self):
        return str(self.__dict__)

    def _single_session_fit(self, vols, n_jobs=1, quality=None,
                            affine_correction=None):
        """
        Realigns volumes (vols) from a single session.

        Parameters
        ----------
        vols: list of `nibabel.Nifti1Image` objects
            the volumes to realign

        quality: float, optional (default None)
            to override instance value

        affine_correction: 2D array of shape (4, 4), optional (default None)
            affine transformation to be applied to vols before realignment
            (this is useful in multi-session realignment)

        Returns
        -------
        2D array of shape (n_scans, len(self.lkp)
            the estimated realignment parameters

        """

        # sanitize input
        if quality is None:
            quality = self.quality

        if affine_correction is None:
            affine_correction = np.eye(4)

        # load first vol
        vols = load_vols(vols)
        n_scans = len(vols)
        vol_0 = vols[0]

        # single vol ?
        if n_scans < 2:
            return np.array([get_initial_motion_params()])

        # affine correction
        vol_0 = nibabel.Nifti1Image(
            vol_0.get_data(), np.dot(affine_correction, vol_0.get_affine()))

        # voxel dimensions on the working grid
        skip = np.sqrt(np.sum(vol_0.get_affine()[:3, :3] ** 2, axis=0)
                       ) ** (-1) * self.sep

        # build working grid
        dim = vol_0.shape
        x1, x2, x3 = np.mgrid[0:dim[0] - .5 - 1:skip[0],
                              0:dim[1] - .5 - 1:skip[1],
                              0:dim[2] - .5 - 1:skip[2]].reshape((3, -1))

        # smooth 0th volume to absorb noise before differentiating
        sref_vol = self.smooth_func(vol_0, self.fwhm).get_data()

        # resample the smoothed reference volume unto doped working grid
        G = ndimage.map_coordinates(sref_vol, [x1, x2, x3], order=self.interp,
                                    mode='wrap',).reshape(x1.shape)

        # compute gradient of reference volume
        Gx, Gy, Gz = np.gradient(sref_vol)

        # resample gradient unto working grid
        Gx = ndimage.map_coordinates(Gx, [x1, x2, x3], order=self.interp,
                                     mode='wrap',).reshape(x1.shape)
        Gy = ndimage.map_coordinates(Gy, [x1, x2, x3], order=self.interp,
                                     mode='wrap',).reshape(x1.shape)
        Gz = ndimage.map_coordinates(Gz, [x1, x2, x3], order=self.interp,
                                     mode='wrap',).reshape(x1.shape)

        # compute rate of change of chi2 w.r.t. parameters
        A0 = _compute_rate_of_change_of_chisq(vol_0.get_affine(),
                                              [x1, x2, x3], [Gx, Gy, Gz],
                                              lkp=self.lkp)

        # compute intercept vector for LSPs
        b = G.ravel()

        # garbage-collect unused local variables
        del sref_vol, Gx, Gy, Gz

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
                for t in range(A0.shape[0]):
                    tmp = np.hstack((A0[t, ...], b[t])).reshape(
                        (1, A0.shape[1] + 1))
                    dets[t] = scipy.linalg.det(alpha - np.dot(tmp.T, tmp))
                msk = np.argsort(det1 - dets)
                msk = msk[:int(np.round(len(dets) / 10.))]

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

                # updates for next iteration
                alpha = np.vstack((A0.T, b)).T
                alpha = np.dot(alpha.T, alpha)
                det1 = scipy.linalg.det(alpha)
            self._log(
                "...done; eliminated %i voxels.\r\n" % n_eliminated_voxels)

        # register the volumes to the reference volume
        self._log("Registering volumes to reference ( = volume 1)...")
        rp = np.ndarray((n_scans, 12))
        rp[0, ...] = get_initial_motion_params(
            )  # don't mov the reference image

        if n_jobs > 1:
            svf_kwargs = {}
        else:
            svf_kwargs = {'log': self._log}

        rps = Parallel(n_jobs=n_jobs)(delayed(
              _single_volume_fit)(vol, vol_0.get_affine(), A0,
                                  affine_correction,
                                  b, x1, x2, x3, fwhm=self.fwhm,
                                  n_iterations=self.n_iterations,
                                  interp=self.interp, lkp=self.lkp,
                                  tol=self.tol, smooth_func=self.smooth_func,
                                  **svf_kwargs) for vol in vols[1:])
        rp[1:, ...] = np.array(rps)

        return rp

    def fit(self, vols, n_jobs=1):
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

        n_jobs: int
            number of parallel jobs

        Returns
        -------
        `MRIMotionCorrection` instance
            fitted object

        Raises
        ------
        RuntimeError, ValueError

        """

        # sanitize vols and n_sessions
        if isinstance(vols, str) or isinstance(
                vols, nibabel.Nifti1Image):
            vols = [vols]
        if len(vols) != self.n_sessions:
            raise RuntimeError(
                "Number of session volumes (%i) != number of sessions (%i)"
                % (len(vols), self.n_sessions))

        self.vols_ = vols
        self.vols = [vols] if isinstance(vols, str) else list(vols)
        if len(self.vols) != self.n_sessions:
            if self.n_sessions == 1:
                self.vols = [self.vols]
            else:
                raise ValueError

        # load first vol of each session
        first_vols = []
        n_scans_list = []
        for sess in range(self.n_sessions):
            try:
                self.vols_[sess] = load_vols(self.vols_[sess])
            except ValueError:
                pass
            vol_0 = self.vols_[sess][0]
            n_scans = len(self.vols_[sess])
            first_vols.append(vol_0)
            n_scans_list.append(n_scans)
        self.n_scans_list = n_scans_list

        # realign first vol of each session with first vol of first session
        if self.n_sessions > 1:
            self._log(
                ('\r\nInter-session registration: realigning first volumes'
                 ' of all sessions...'))

        self.first_vols_realignment_parameters_ = self._single_session_fit(
            first_vols,
            quality=1.  # only a few vols, we can thus allow this lux
            )

        rfirst_vols = apply_realignment(
            first_vols, self.first_vols_realignment_parameters_, inverse=False)

        if self.n_sessions > 1:
            self._log('...done (inter-session registration).\r\n')

        # realign all vols of each session with first vol of that session
        self.realignment_parameters_ = []
        for sess in range(self.n_sessions):
            self._log(
                ("Intra-session registration: Realigning session"
                 " %i/%i...") % (sess + 1, self.n_sessions))

            # affine correction, for inter-session realignment
            affine_correction = np.dot(rfirst_vols[sess].get_affine(),
                                       scipy.linalg.inv(
                                           rfirst_vols[0].get_affine()))

            sess_rp = self._single_session_fit(
                self.vols_[sess],
                affine_correction=affine_correction,
                n_jobs=n_jobs)

            self.realignment_parameters_.append(sess_rp)
            self._log('...done; session %i.\r\n' % (sess + 1))

        # beware, the clumpsy list comprehension is because sessions may have
        # different number of volumes (see issue #36, for example)
        self.realignment_parameters_ = [
            sess_rp[..., :6]
            for sess_rp in self.realignment_parameters_]

        return self

    def transform(self, output_dir=None, reslice=False, prefix="r",
                  basenames=None, ext=None, concat=False):
        """
        Saves realigned volumes and the realigment parameters to disk.
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
            file extension for ouput images; can be ".img", ".nii", or
            ".nii.gz"

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
            realigned_images: list of strings
                full paths of the realigned files

            realignment_parameters_: string
                full patsh of text file containing realignment parameters
        """
        # make sure object has been fitted
        if not hasattr(self, 'realignment_parameters_'):
            raise RuntimeError("fit(...) method not yet invoked.")

        # sanitize reslice param
        reslice = reslice or concat  # can't conct without reslicing

        # output dict
        output = {"realigned_images": [],
                  "realignment_parameters": self.realignment_parameters_
                  if output_dir is None else []}

        for sess in range(self.n_sessions):
            concat_sess = concat
            if (isinstance(self.vols_[sess], str) or is_niimg(
                    self.vols_[sess])) and reslice:
                concat_sess = True

            n_scans = len(self.realignment_parameters_[sess])
            sess_realigned_files = []

            # modify the header of each 3D vol according to the
            # estimated motion (realignment params)
            sess_rvols = apply_realignment(self.vols_[sess],
                                           self.realignment_parameters_[sess],
                                           inverse=False)

            # reslice vols
            if reslice:
                self._log('Reslicing volumes for session %i/%i...' % (
                    sess + 1, self.n_sessions))
                sess_rvols = list(reslice_vols(sess_rvols))
                self._log('...done; session %i/%i.' % (
                    sess + 1, self.n_sessions))

            if concat_sess:
                sess_rvols = nibabel.concat_images(sess_rvols)

            if output_dir is None:
                output['realigned_images'].append(sess_rvols)

            # save output unto disk
            if not output_dir is None:
                # make basenames for output files
                sess_basenames = None
                if basenames is None:
                    if isinstance(self.vols[sess], str):
                        sess_basenames = get_basenames(self.vols[sess],
                                                       ext=ext)
                    elif isinstance(self.vols[sess], list):
                        if isinstance(self.vols[sess][0], str):
                            sess_basenames = get_basenames(self.vols[sess],
                                                           ext=ext)
                    else:
                        if not isinstance(self.vols, list) or concat:
                            sess_basenames = "vols"

                        else:
                            sess_basenames = ["sess_%i_vol_%i" % (sess, i)
                                              for i in range(n_scans)]
                else:
                    assert len(basenames) == self.n_sessions
                    sess_basenames = basenames[sess]

                # save realigned files to disk
                if concat_sess:
                    sess_realigned_files = save_vols(
                        sess_rvols,
                        output_dir,
                        basenames=sess_basenames if isinstance(
                            sess_basenames, str)
                        else sess_basenames[0], concat=concat_sess, ext=ext,
                        prefix=prefix)
                else:
                    sess_realigned_files = [save_vol(
                        sess_rvols[t],
                        output_dir=output_dir,
                        basename=sess_basenames[t] if isinstance(
                            sess_basenames, list) else "vol_%i" % t,
                        ext=ext, prefix=prefix) for t in range(n_scans)]

                output['realigned_images'].append(sess_realigned_files)

                # save realignment params to disk
                if basenames is None:
                    sess_realignment_parameters_filename = os.path.join(
                        output_dir, "rp.txt")
                else:
                    if isinstance(sess_basenames, str):
                        sess_realignment_parameters_filename = os.path.join(
                            output_dir,
                            "rp_" + sess_basenames + ".txt")
                    else:
                        sess_realignment_parameters_filename = os.path.join(
                            output_dir,
                            "rp_" + sess_basenames[0] + ".txt")

                np.savetxt(sess_realignment_parameters_filename,
                           self.realignment_parameters_[sess][..., self.lkp])
                output['realignment_parameters'].append(
                    sess_realignment_parameters_filename)

                self._log('...done; output saved to %s.' % output_dir)

        # return
        return output
