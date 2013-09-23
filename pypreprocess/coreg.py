"""
:Module: coreg.py
:Synopsis: Histogram-based co-registration of 3D MRI images.
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

import scipy.ndimage
import scipy.optimize
import scipy.special
import scipy.signal
import numpy as np
import nibabel
from .affine_transformations import (spm_matrix,
                                     apply_realignment_to_vol,
                                     nibabel2spm_affine
                                     )
from .io_utils import (loaduint8,
                       get_basenames,
                       save_vols,
                       load_specific_vol
                       )
from .kernel_smooth import (fwhm2sigma,
                            centered_smoothing_kernel
                            )
from .histograms import (trilinear_interp,
                         joint_histogram,
                         _correct_voxel_samp,
                         make_sampled_grid
                         )

# 'texture' of floats in machine precision
EPS = np.finfo(float).eps


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
        krn1 = centered_smoothing_kernel(fwhm[0],
                                         np.linspace(-1 * lim[0], lim[0],
                                                      num=2 * lim[0]))
        krn1 = krn1 / np.sum(krn1)
        krn2 = centered_smoothing_kernel(fwhm[1],
                                         np.linspace(-1 * lim[1], lim[1],
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

    Examples
    --------
    >>> from pypreprocess.coreg import Coregister
    >>> c = Coregister()
    >>> ref = '/home/elvis/CODE/datasets/spm_auditory/sM00223/sM00223_002.img'
    >>> src = '/home/elvis/CODE/datasets/spm_auditory/fM00223/fM00223_004.img'
    >>> c.fit(ref, src)
    >>> c.transform(src)

    References
    ----------
    [1] Rigid Body Registration, by J. Ashburner and K. Friston

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

    def __repr__(self):
        return str(self.__dict__)

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
        coregistered_source = apply_realignment_to_vol(src, self.params_,
                                                       inverse=True
                                                       )

        # save output unto disk
        if not output_dir is None:
            if isinstance(src, basestring):
                basenames = get_basenames(src)
            else:
                basenames = 'coregistered_source'

            coregistered_source = save_vols(coregistered_source,
                                            output_dir=output_dir,
                                            basenames=basenames,
                                            ext=ext, prefix=prefix)

        return coregistered_source
