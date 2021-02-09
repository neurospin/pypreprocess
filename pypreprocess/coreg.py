"""
:Module: coreg.py
:Synopsis: Histogram-based co-registration of 3D MRI images.
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

from scipy.ndimage import gaussian_filter
from scipy.optimize import fmin_powell
import scipy.special
from scipy.signal import sepfir2d
import numpy as np
import nibabel
from .affine_transformations import (spm_matrix, apply_realignment,
                                     nibabel2spm_affine)
from .io_utils import loaduint8, get_basenames, save_vols
from .kernel_smooth import fwhm2sigma, centered_smoothing_kernel
from .histograms import (trilinear_interp, joint_histogram,
                         _correct_voxel_samp, make_sampled_grid)

# 'texture' of floats in machine precision
EPS = np.finfo(float).eps


def compute_similarity_from_jhist(jh, fwhm=None, cost_fun='nmi'):
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
    if fwhm is None:
        fwhm = [7., 7.]
    if len(np.shape(jh)) != 2:
        raise ValueError("jh must be 2D array, got %s" % jh)

    if len(np.shape(fwhm)) == 0:
        fwhm = [fwhm] * 2
    fwhm = fwhm[:2]

    # create separable filter for smoothing the joint-histogram
    lim = np.ceil(fwhm * 2).astype(np.int)
    krn1 = centered_smoothing_kernel(fwhm[0],
                                     np.linspace(-1 * lim[0], lim[0],
                                                 num=2 * lim[0]))
    # Pad with zero to get odd length
    krn1 = np.append(krn1, 0)
    krn1 = krn1 / np.sum(krn1)
    krn2 = centered_smoothing_kernel(fwhm[1],
                                     np.linspace(-1 * lim[1], lim[1],
                                                 num=2 * lim[1]))
    # Pad with zero to get odd length
    krn2 = np.append(krn2, 0)
    krn2 = krn2 / np.sum(krn2)

    # smooth the histogram with kern1 x kern2
    jh = sepfir2d(jh, krn1, krn2)

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
        nmi = (np.sum(s1 * np.log2(s1)) + np.sum(s2 * np.log2(s2))) / np.sum(
            np.sum(jh * np.log2(jh)))
        o = -nmi
    else:
        raise NotImplementedError(
            "Unsupported cost_fun (cost function): %s" % cost_fun)

    return o


def compute_similarity(params, ref, src, ref_affine, src_affine, grid,
                       cost_fun='nmi', fwhm=None, bins=(256, 256)):
    """
    Computes the similarity between the reference image (ref) and the moving
    image src, under the current affine motion parameters (x).

    The computed similarity measure is histogram-based.

    Parameters
    ----------
    params: 1D array of 6 floats
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
    if fwhm is None:
        fwhm = [7., 7.]

    # compute affine transformation matrix
    params = np.array(params)
    M = np.dot(scipy.linalg.lstsq(src_affine,
                                  spm_matrix(params))[0],
               ref_affine)

    # create the joint histogram
    jh = joint_histogram(ref.copy(), src.get_data(), grid=grid, M=M, bins=bins)

    # compute similarity from joint histgram
    return compute_similarity_from_jhist(jh, fwhm=fwhm, cost_fun=cost_fun)


def _run_powell(params, direct, tolsc, *otherargs):
    """
    Run Powell optimization.

    Parameters
    ----------
    params: 1D array of 6 floats
        starting estimates for realignment parameters

    direc: 2D array of shape (n_directions, 6)
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
        Run compute_similarity API with verbose reporting.

        """

        output = compute_similarity(x, *otherargs)

        # verbose
        token = "".join(['%-12.4g ' % z for z in x])
        token += '|  %.5g' % output
        print(token)

        return output

    # fire!
    return fmin_powell(_compute_similarity, params,
                       direc=direct,
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

    References
    ----------
    [1] Rigid Body Registration, by J. Ashburner and K. Friston

    """

    def __init__(self,
                 sep=np.array([4, 2]),
                 params_init=np.zeros(6),
                 tol=np.array([.02, .02, .02, .001, .001, .001]),
                 cost_fun="nmi",
                 smooth_vols=True,
                 fwhm=np.array([7., 7., 7.]),
                 bins=(256, 256),
                 verbose=1
                 ):
        self.sep = sep
        self.params_init = params_init
        self.tol = tol
        self.cost_fun = cost_fun
        self.fwhm = fwhm
        self.bins = bins
        self.smooth_vols = smooth_vols
        self.verbose = verbose

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

    def fit(self, target, source):
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
        `Coregistration` instance
            fitted object
        """
        # configure the Powell optimizer
        self.sc_ = np.array(self.tol)
        self.sc = self.sc_[:len(self.params_init)]
        self.search_direction_ = np.diag(self.sc_ * 20)

        # load vols
        target = loaduint8(target)
        source = loaduint8(source)

        # tweak affines so we can play SPM games everafter
        target = nibabel.Nifti1Image(target.get_data(),
                                     nibabel2spm_affine(target.get_affine()))
        source = nibabel.Nifti1Image(source.get_data(),
                                     nibabel2spm_affine(source.get_affine()))

        # smooth images according to pyramidal sep
        if self.smooth_vols:
            # target
            vxg = np.sqrt(np.sum(target.get_affine()[:3, :3] ** 2, axis=0))
            fwhmg = np.sqrt(np.maximum(
                np.ones(3) * self.sep[-1] ** 2 - vxg ** 2,
                [0, 0, 0])) / vxg
            target = nibabel.Nifti1Image(
                gaussian_filter(target.get_data(),
                                fwhm2sigma(fwhmg)),
                target.get_affine())

            # source
            vxf = np.sqrt(np.sum(source.get_affine()[:3, :3] ** 2, axis=0))
            fwhmf = np.sqrt(np.maximum(
                np.ones(3) * self.sep[-1] ** 2 - vxf ** 2,
                [0, 0, 0])) / vxf
            source = nibabel.Nifti1Image(gaussian_filter(
                source.get_data(), fwhm2sigma(fwhmf)), source.get_affine())

        # pyramidal loop
        self.params_ = np.array(self.params_init)
        for samp in self.sep:
            print("\r\nRunning Powell gradient-less local optimization "
                  "(pyramidal level = %smm)..." % samp)

            # create sampled grid for target img
            grid = make_sampled_grid(target.shape, samp=_correct_voxel_samp(
                target.get_affine(), samp))

            # interpolate target on sampled grid
            sampled_target = trilinear_interp(
                target.get_data().ravel(order='F'),
                target.shape, *grid)

            # find optimal realignment parameters
            self.params_ = _run_powell(
                self.params_, self.search_direction_, self.sc_,
                sampled_target, source, target.get_affine(),
                source.get_affine(), grid, self.cost_fun,
                self.fwhm, self.bins)

        return self

    def transform(self, source, output_dir=None, prefix="", ext=".nii.gz",
                  basenames=None):
        """
        Applies estimated co-registration parameter to the input volume
        (source).

        Parameters
        ----------
        source: string (existing filename) or 3D nibabel image object
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
        # save output unto disk
        if output_dir is not None:
            if basenames is None:
                basenames = get_basenames(source)

        # apply coreg
        # XXX backend should handle nasty i/o logic!!
        coregistered_source = list(apply_realignment(source, self.params_,
                                                     inverse=True))
        if output_dir is not None:
            concat = isinstance(basenames, str)
            coregistered_source = save_vols(coregistered_source,
                                            output_dir=output_dir,
                                            basenames=basenames,
                                            ext=ext, prefix=prefix,
                                            concat=concat)
        return coregistered_source
