"""
:Module: spm_realign
:Synopsis: MRI within-modality Motion Correction, SPM style
:Author: DOHMATOB Elvis Dopgima

"""

import nibabel as ni
import glob
import os
import numpy as np
import scipy.ndimage as ndi
import scipy.linalg
import itertools
import kernel_smooth
import affine_transformations


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
        x1, x2, x3 = zip(*itertools.product(x1, x2, x3))
    else:
        x1 = x1.ravel()
        x2 = x2.ravel()
        x3 = x3.ravel()

    A = np.ndarray((len(x1), len(lkp)))

    # loop over parameters (this loop computes columns of A)
    for i in xrange(len(lkp)):
        pt = affine_transformations.get_initial_motion_params()

        # regularization ith parameter
        pt[lkp[i]] = pt[i] + 1e-6

        # map cartesian coordinate space according to motion
        # parameters pt (using jacobian associated with the transformation)
        y1, y2, y3  = affine_transformations.coords(pt, M, M, x1, x2, x3)

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


class MRIMotionCorrection(object):
    """Implements within-modality rigid-body registration of MRI volumes.

    The fit(...) method  estimates affine transformations necessary in other to
    rigidly align the other volumes to the Oth volume (hereafter referred to as
    the reference), whilst the transform(...) method actually writes these
    realigned files unto disk.

    """

    def __init__(self, sep=4, interp=3, fwhm=5., quality=.9, rtm=False,
                 lkp=[0, 1, 2, 3, 4, 5], PW=None, verbose=1,
                 output_dir=None,
                 prefix='r',
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
        sep: intx, optional (default 4)
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
        output_dir: string, optional (default None)
            existing dirname, where all output (i.e realigned files) will be
            written
        prefix: string, optional (default 'r')
            prefix to be prepended to output file basenames

        """

        self._sep = sep
        self._interp = interp
        self._fwhm = fwhm
        self._quality = quality
        self._rtm = rtm
        self._lkp = lkp
        self._PW = PW
        self._verbose = verbose
        self._output_dir = output_dir
        self._prefix = prefix

        # check against unimplemented option rtm
        if self._rtm:
            raise NotImplementedError("rtm: Option not yet implemented")

        # check against unimplemented option PW
        if self._PW:
            raise NotImplementedError("PW: Option not yet implemented")

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
        """Estimation of within-modality rigid-body movement parameters.

        All operations are performed relative to the first image. That is,
        registration is to the first image, and resampling of images is
        into the space of the first image.

        The algorithm is a Gauss-Netwon iterative refinememnt of an l2 loss
        (squared difference between images)

        Parameters
        ----------
        filenames: string or list of strings
            filename of single 4D image, or list of filenames of 3D volumes

        Returns
        -------
        rp: 2D array of shape (n_scans, 6)
            realignment parameters

        Notes
        -----
        For now, we don't have support for multiple sessions!

        Raises
        ------
        RuntimeError, ValueError

        """

        # load data
        self._filenames = filenames if len(filenames) > 1 else filenames[0]
        if isinstance(self._filenames, basestring):
            film = ni.load(self._filenames)
            if not len(film.shape) == 4:
                raise ValueError(
                    "Expecting 4D image, got %iD!" % len(film.shape))
            self._P = [ni.Nifti1Image(film.get_data()[..., t],
                                      film.get_affine())
                       for t in xrange(film.shape[-1])]
        else:
            self._P = [ni.load(filename) for filename in self._filenames]

        # voxel dimensions on the working grid
        skip = np.sqrt(np.sum(self._P[0].get_affine()[:3, :3] ** 2, axis=0)
                       ) ** (-1) * self._sep

        # aus variable of volume shape
        dim = self._P[0].shape[:3]

        # build working grid
        x1, x2, x3 = np.mgrid[:dim[0] - .5 - 1:skip[0],
                               :dim[1] - .5 - 1:skip[1],
                               :dim[2] - .5 - 1:skip[2]]

        # dope the grid so we're not perfectly aligned with its vertices
        x1 += np.random.randn(*x1.shape) * .5
        x2 += np.random.randn(*x2.shape) * .5
        x3 += np.random.randn(*x3.shape) * .5

        # ravel working grid
        x1 = x1.ravel()
        x2 = x2.ravel()
        x3 = x3.ravel()

        # smooth 0th volume to absorb noise before differentiating
        sref_vol = kernel_smooth.smooth_image(self._P[0],
                                              self._fwhm).get_data()

        # resample the smoothed reference volume unto doped working grid
        G = ndi.map_coordinates(sref_vol, [x1.ravel(), x2.ravel(), x3.ravel()],
                                order=self._interp,
                                mode='wrap',).reshape(x1.shape)

        # compute gradient of reference volume
        Gx, Gy, Gz = np.gradient(sref_vol)

        # resample gradient unto working grid
        Gx = ndi.map_coordinates(Gx, [x1.ravel(), x2.ravel(), x3.ravel()],
                                order=self._interp,
                                mode='wrap',).reshape(x1.shape)
        Gy = ndi.map_coordinates(Gy, [x1.ravel(), x2.ravel(), x3.ravel()],
                                order=self._interp,
                                mode='wrap',).reshape(x1.shape)
        Gz = ndi.map_coordinates(Gz, [x1.ravel(), x2.ravel(), x3.ravel()],
                                order=self._interp,
                                mode='wrap',).reshape(x1.shape)

        # compute rate of change of chi2 w.r.t. parameters
        A0 = make_A(self._P[0].get_affine(), x1, x2, x3, Gx, Gy, Gz,
                )

        # ravel everything
        G = G.ravel()
        Gx = Gx.ravel()
        Gy = Gy.ravel()
        Gz = Gz.ravel()

        b = G

        # Remove voxels that contribute very little to the final estimate.
        # It's essentially sufficient to remove voxels that contribute the
        # least to the determinant of the inverse convariance matrix
        # (i.e of the precision matrix)
        self._log(
            ("\r\nEliminating unimportant voxels (target quality: %s)"
             "...") % self._quality)
        alpha = np.vstack((A0.T, b)).T
        alpha = np.dot(alpha.T, alpha)
        det0 = scipy.linalg.det(alpha)
        det1 = det0  # det1 / det0 is a precision measure
        iteration = 0
        while det1 / det0 > self._quality:
            iteration += 1

            # determine unimportant voxels to eliminate
            dets = np.ndarray(A0.shape[0])
            for t in xrange(A0.shape[0]):
                tmp = np.hstack((A0[t, ...], b[t])).reshape((1,
                                                             A0.shape[1] + 1))
                dets[t] = scipy.linalg.det(alpha - np.dot(tmp.T, tmp))
            msk = np.argsort(det1 - dets)
            msk = msk[:np.round(len(dets) / 10.)]

            # eliminate unimportant voxels
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

        # register the volumes to the reference volume
        self._log("\r\nRegistering volumes to reference...")
        for t in xrange(1, len(self._P)):
            self._log("\tRegistering volume %i/%i..." % (t, len(self._P) - 1))

            # smooth tth volume
            V = kernel_smooth.smooth_image(self._P[t], self._fwhm).get_data()

            # Run Gauss-Newton Iterated LS (this should normally converge after
            # about as few as 5 iterations)
            dim = np.hstack((V.shape, [1, 1]))
            dim = dim[:3]
            ss = -np.inf
            countdown = -1
            for iteration in xrange(64):
                # pass from tth volume's grid to that of the reference
                # volume (0)
                y1, y2, y3 = affine_transformations.coords(
                    [0, 0, 0, 0, 0, 0],
                    self._P[0].get_affine(),
                    self._P[t].get_affine(), x1, x2, x3)

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

                # warp: resample tth volume on this new grid
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
                p = affine_transformations.get_initial_motion_params()
                p[self._lkp] += soln.T

                # apply the learnt affine transformation to tth volume
                self._P[t] = ni.Nifti1Image(
                    self._P[t].get_data(),
                    np.dot(scipy.linalg.inv(
                            affine_transformations.spm_matrix(p)),
                           self._P[t].get_affine()))

                # compute convergence criterion variables
                pss = ss
                ss = np.sum(b1 ** 2) / len(b1)

                # compute relative gain over last iteration
                relative_gain = np.abs((pss - ss) / pss)
                self._log(
                    "\t\trelative gain over last iteration: %s" % relative_gain
                    )

                # check whether we've stopped converging altogether
                if relative_gain < 1e-8 and countdown == -1:
                    countdown = 2

                # countdown
                if countdown != -1:
                    # converge
                    if countdown == 0:
                        break
                    countdown -= 1

        # extract the estimated motion params from the final affines
        self._rp = np.ndarray((len(self._P), 12))
        iref_affine = scipy.linalg.inv(self._P[0].get_affine())
        for t in xrange(len(self._P)):
            self._rp[t, ...] = affine_transformations.spm_imatrix(np.dot(
                    self._P[t].get_affine(), iref_affine))

        # return estimated motion (realignment parameters)
        return self._rp

    def transform(self, output_dir=None):
        """Saves realigned volumes and the realigment parameters to disk.

        The input volumes were as specified in a single filename, then
        the output realigned volumes will have basename of the form
        rmy_4D_basename_volume_t.ext, where 'r' is the prefix, my_4D_basename
        is the input file basename, t is the index of the specific a
        volume, and ext is the extension (.nii, .nii.gz, .img, etc.).
        Otherwise, the realigned volumes will be stored in files
        rmy_3D_volume_t.ext, where 'r' is the extension, my_3D_volume_t is
        if the basename of the corresponding input 3D volume for time scan
        number t, and ext is the extension.

        Parameters
        ----------
        output_dir: string, optional (default None)
            existing dirname where output will be written

        """

        # make sure fit(...) method has been invoked
        if not hasattr(self, '_filenames') or not hasattr(self, '_P'):
            raise RuntimeError(
                "fit(...) method not yet invoked; nothing to do!")

        # sanitize output_dir
        ref_filename = self._filenames if isinstance(
            self._filenames, basestring) else self._filenames[0]
        if self._output_dir is None:
            self._output_dir = output_dir
        if self._output_dir is None:
            self._output_dir = os.path.dirname(ref_filename)
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        # save realignment parameters to disk
        rp_filename = os.path.join(self._output_dir,
                                   "rp_%s.txt" % os.path.basename(
                ref_filename).split('.')[0])
        np.savetxt(rp_filename, self._rp[..., :6])

        # save realigned files to disk
        self._log('\r\nSaving realigned volumes to %s' % self._output_dir)
        if isinstance(self._filenames, basestring):
            for t in xrange(len(self._P)):
                _parts = os.path.basename(
                    ref_filename).split('.')
                ref_filebasname, ext = _parts[0], ".".join(_parts[1:])
                output_filename = os.path.join(self._output_dir,
                                               "%s%s_volume_%i.%s" % (
                        self._prefix, ref_filebasname, t, ext))
                ni.save(self._P[t], output_filename)
        else:
            for filename, vol in zip(self._filenames, self._P):
                output_filename = os.path.join(self._output_dir,
                                               "%s%s" % (
                        self._prefix,
                        os.path.basename(filename)))
                ni.save(vol, output_filename)


def _demo(subjects, dataset_id):
    """Demo runner.

    Parameters
    ----------
    subjects: iterable for subject data
        each subject data can be anything, with a func (string or list
        of strings; existing file path(s)) and an output_dir (string,
        existing dir path) field
    dataset_id: string
        a short string describing the data being processed (e.g. "HAXBY!")

    Notes
    -----
    Don't invoke this directly!

    """

    from reporting.check_preprocessing import plot_spm_motion_parameters
    import matplotlib.pyplot as plt

    # loop over subjects
    for subject_data in subjects:
        print("\t\tMotion correction for %s" % subject_data.subject_id)

        # instantiate realigner
        mc = MRIMotionCorrection(fwhm=5., interp=2)

        # fit realigner
        mc.fit(subject_data.func)

        # write realigned files to disk
        mc.transform(output_dir=subject_data.output_dir)

        # plot results
        rp_filename = glob.glob(os.path.join(subject_data.output_dir,
                                             "rp_*.txt"))[0]
        plot_spm_motion_parameters(
            rp_filename,
            title="Estimated motion for %s of '%s'" % (
                subject_data.subject_id,
                dataset_id))
        plt.show()


def demo_nyu_rest(DATA_DIR="/tmp/nyu_data",
                  OUTPUT_DIR="/tmp/nyu_mrimc_output",
                  ):
    """Demo for FSL Feeds data.

    Parameters
    ----------
    DATA_DIR: string, optional
        where the data is located on your disk, where it will be
        downloaded to
    OUTPUT_DIR: string, optional
        where output will be written to

    """

    import sys
    sys.path.append(
        os.path.dirname(os.path.dirname(
                os.path.split(os.path.abspath(__file__))[0])))
    from external.nisl.datasets import fetch_nyu_rest

    # fetch data
    nyu_data = fetch_nyu_rest(data_dir=DATA_DIR)

    class SubjectData(object):
        pass

    # subject data factory
    def subject_factory(session=1):
        session_output_dir = os.path.join(OUTPUT_DIR, "session%i" % session)

        session_func = [x for x in nyu_data.func if "session%i" % session in x]
        session_anat = [
            x for x in nyu_data.anat_skull if "session%i" % session in x]

        for subject_id in set([os.path.basename(
                    os.path.dirname
                    (os.path.dirname(x)))
                               for x in session_func]):

            # instantiate subject_data object
            subject_data = SubjectData()
            subject_data.subject_id = subject_id
            subject_data.session_id = session

            # set func
            subject_data.func = [
                x for x in session_func if subject_id in x]
            assert len(subject_data.func) == 1
            subject_data.func = subject_data.func[0]

            # set anat
            subject_data.anat = [
                x for x in session_anat if subject_id in x]
            assert len(subject_data.anat) == 1
            subject_data.anat = subject_data.anat[0]

            # set subject output directory
            subject_data.output_dir = os.path.join(
                session_output_dir, subject_data.subject_id)

            yield subject_data

    # invoke demon to run de demo
    _demo(subject_factory(), "NYU Resting State")


def demo_fsl_feeds(DATA_DIR="/tmp/fsl_feeds_data",
                  OUTPUT_DIR="/tmp/fsl_feeds_mrimc_output",
                  ):
    """Demo for FSL Feeds data.

    Parameters
    ----------
    DATA_DIR: string, optional
        where the data is located on your disk, where it will be
        downloaded to
    OUTPUT_DIR: string, optional
        where output will be written to

    """

    import sys
    sys.path.append(
        os.path.dirname(os.path.dirname(
                os.path.split(os.path.abspath(__file__))[0])))
    from datasets_extras import fetch_fsl_feeds_data
    from reporting.check_preprocessing import plot_spm_motion_parameters

    # fetch data
    fsl_feeds_data = fetch_fsl_feeds_data(data_dir=DATA_DIR)

    class SubjectData(object):
        pass

    # subject data factory
    def subject_factory(session=1):
            # instantiate subject_data object
            subject_data = SubjectData()
            subject_data.subject_id = "sub001"

            # set func
            subject_data.func = fsl_feeds_data['func'] + ".gz"

            # set subject output directory
            subject_data.output_dir = OUTPUT_DIR

            yield subject_data

    # invoke demon to run de demo
    _demo(subject_factory(), "FSL FEEDS")


# main
if __name__ == '__main__':
    import sys
    warning = ("%s: THIS SCRIPT MUST BE RUN FROM ITS PARENT "
               "DIRECTORY!") % sys.argv[0]
    banner = "#" * len(warning)
    separator = "\r\n\t"

    print separator.join(['', banner, warning, banner, ''])

    # run FSL FEEDS demo
    demo_fsl_feeds()
