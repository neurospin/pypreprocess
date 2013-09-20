"""
:Author: DOHMATOB Elvis Dopgima

"""

import os
from collections import namedtuple
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.special
import scipy.optimize
import scipy.io
import nibabel
import spm_hist2py

from .io_utils import (is_niimg,
                       save_vol,
                       get_basenames,
                       load_specific_vol,
                       loaduint8
                       )
from .affine_transformations import (spm_matrix,
                                     nibabel2spm_affine
                                     )
from .kernel_smooth import fwhm2sigma

# flags for fitting coregistration model
Flags = namedtuple('Flags', 'fwhm sep cost_fun tol params')

# 'texture' of floats in machine presicion
EPS = np.finfo(float).eps


def smoothing_kernel(fwhm, x):
    # variance from fwhm
    s = fwhm ** 2 / (8 * np.log(2)) + EPS

    # Gaussian convolve with 0th degree B-spline
    w1 = .5 * np.sqrt(2 / s)
    w2 = -.5 / s
    w3 = np.sqrt(s / 2 / np.pi)
    krn = .5 * (scipy.special.erf(w1 * (x + 1)) * (x + 1) + scipy.special.erf(
            w1 * (x - 1)) * (x - 1) - 2 * scipy.special.erf(
            w1 * x) * x) + w3 * (np.exp(w2 * (x + 1) ** 2) + np.exp(
            w2 * (x - 1) ** 2) - 2 * np.exp(w2 * x ** 2))

    krn[krn < 0.] = 0

    return krn


def optfun(x, ref_vol, src_vol, s=[1, 1, 1], cf='mi', fwhm=[7., 7.]):
    """
    Returns
    -------
    o

    """

    x = np.array(x)

    # voxel sizes
    vxg = np.sqrt(np.sum(ref_vol.get_affine()[:3, :3] ** 2, axis=0))
    sg = s / vxg

    # create the joint histogram
    M = np.dot(scipy.linalg.lstsq(src_vol.get_affine(),
                                  spm_matrix(x))[0],
                                  ref_vol.get_affine())
    H = spm_hist2py.hist2py(M, ref_vol.get_data(), src_vol.get_data(), sg)

    # Smooth the histogram
    lim  = np.ceil(fwhm * 2)
    krn1 = smoothing_kernel(fwhm[0], np.linspace(-1 * lim[0], lim[0],
                                                  num=2 * lim[0]))
    krn1 = krn1 / np.sum(krn1)
    krn2 = smoothing_kernel(fwhm[1], np.linspace(-1 * lim[1], lim[1],
                                                  num=2 * lim[1]))
    krn2 = krn2 / np.sum(krn2)

    # H = scipy.signal.sepfir2d(H, krn1, krn2)
    H = scipy.ndimage.gaussian_filter(H,
                                      sigma=fwhm2sigma(fwhm[:2]),
                                      mode='wrap'
                                      )

    # compute marginal histograms
    H = H + EPS
    sh = np.sum(H)
    H = H / sh
    s1 = np.sum(H, axis=0).reshape((-1, H.shape[0]), order='F')
    s2 = np.sum(H, axis=1).reshape((H.shape[1], -1), order='F')

    # compute cost function proper
    if cf == 'mi':
        # Mutual Information:
        H = H * np.log2(H / np.dot(s2, s1))
        mi = np.sum(H)
        o = -mi
    elif cf == 'ecc':
        # Entropy Correlation Coefficient of:
        # Maes, Collignon, Vandermeulen, Marchal & Suetens (1997).
        # "Multimodality image registration by maximisation of mutual
        # information". IEEE Transactions on Medical Imaging 16(2):187-198
        H = H * np.log2(H / np.dot(s2, s1))
        mi = np.sum(H.ravel(order='F'))
        ecc = -2 * mi / (np.sum(s1 * np.log2(s1)) + np.sum(s2 * np.log2(s2)))
        o = -ecc
    elif cf == 'nmi':
        # Normalised Mutual Information of:
        # Studholme,  Hill & Hawkes (1998).
        # "A normalized entropy measure of 3-D medical image alignment".
        # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
        nmi = (np.sum(s1 * np.log2(s1)) + np.sum(
                s2 * np.log2(s2))) / np.sum(np.sum(H * np.log2(H)))
        o = -nmi
    else:
        raise NotImplementedError("Unsupported cf (cost function): %s" % cf)

    return o


def spm_powell(x0, xi, tolsc, *otherargs):
    """
    Run Powell optimization.

    XXX YAGNI: Limit number of wrapper functions, etc.

    """

    def of(x):

        output = optfun(x, *otherargs)

        # verbose
        token = "".join(['%-12.4g ' % z for z in x])
        token += '|  %.5g' % output
        print token

        return output

    def _cb(x):
        print "\r\n\t\tCurrent parameters estimate: %s\r\n" % x

    return scipy.optimize.fmin_powell(of, x0,
                                      direc=xi,
                                      xtol=min(np.min(tolsc), 1e-3),
                                      # callback=_cb
                                      )


def apply_coreg(src_vol, params):
    if isinstance(src_vol, basestring):
        src_vol = nibabel.load(src_vol)
    else:
        assert is_niimg(src_vol)

    return nibabel.Nifti1Image(src_vol.get_data(), scipy.linalg.lstsq(
            spm_matrix(params),
            src_vol.get_affine())[0])


class SPMCoreg(object):
    """
    Similarity-based rigid-body multi-modal registration.

    Parameters
    ----------
    sep: 1D array of floats, optional (default [4, 2])
        piramidal optimization seperation (in mm)
    params: 1D array of length 6, optional (default [0, 0, 0, 0, 0, 0]
        starting estimates
    cost_fun: string, optional (default "nmi")
        similarity function to be optimized. Possible values are:
        "mi": Mutual Information
        "nmi": Normalized Mutual Information
        "ecc": Entropy Correlation Coefficient
    tol: 1D array of 6 floats, optional (
    default [.02, .02, .02, .001, .001, .001])
        tolerances for the accuracy of each parameter

    Notes
    -----
    If your images of the same modality (both fMRI, both PET, etc.), then use
    `MRIMotionCorrection` (from spm_realign module) instead

    """

    def __init__(self,
                 sep=[4, 2],
                 params=np.zeros(6),
                 tol=[.02, .02, .02, .001, .001, .001],
                 cost_fun="nmi",
                 fwhm=[7., 7., 7.],
                 smooth_vols=True,
                 verbose=1
                 ):

        self.sep = np.array(sep)
        self.initial_params = np.array(params)
        self.tol = np.array(tol)
        self.cost_fun = cost_fun
        self.fwhm = np.array(fwhm)
        self.smooth_vols = smooth_vols
        self.verbose = verbose

        # get ready for spm_powell
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

    def fit(self, ref_vol, src_vol):
        """
        Estimates the co-registration parameters for rigidly registering
        src_vol to ref_vol, which can be of different modality
        (fMRI and PET, etc.).

        The technology used is histogram-based registration.

        Parameters
        ----------
        ref_vol: string (existing filename) or 3D (if 4D then then first volume
        will be used) nibabel image object
            reference volume (image that is kept fixed)
        src_vol: string (existing filename) or 3D nibabel image object
            source image (image that is jiggled about to fit the
            reference image)

        Returns
        -------
        Returns
        -------
        `SPMCoreg` instance
            fitted object

        """

        # load ref_vol
        if isinstance(ref_vol, tuple):
            ref_vol = nibabel.Nifti1Image(ref_vol[0], ref_vol[1])
        ref_vol = loaduint8(ref_vol, log=self._log)
        ref_vol = nibabel.Nifti1Image(ref_vol.get_data(),
                                      nibabel2spm_affine(ref_vol.get_affine()))

        # load src_vol
        if isinstance(src_vol, tuple):
            src_vol = nibabel.Nifti1Image(src_vol[0], src_vol[1])

        src_vol = loaduint8(src_vol, log=self._log)
        src_vol = nibabel.Nifti1Image(src_vol.get_data(),
                                      nibabel2spm_affine(src_vol.get_affine()))

        # smooth vols
        if self.smooth_vols:
            vxg = np.sqrt(np.sum(ref_vol.get_affine()[:3, :3] ** 2, axis=0))
            fwhmg = np.sqrt(np.maximum(
                    np.ones(3) * self.sep[-1] ** 2 - vxg ** 2,
                    [0, 0, 0])) / vxg
            ref_vol = nibabel.Nifti1Image(
                scipy.ndimage.gaussian_filter(ref_vol.get_data(),
                                              fwhm2sigma(fwhmg)),
                ref_vol.get_affine())

            vxf = np.sqrt(np.sum(src_vol.get_affine()[:3, :3] ** 2, axis=0))
            fwhmf = np.sqrt(np.maximum(
                    np.ones(3) * self.sep[-1] ** 2 - vxf ** 2,
                    [0, 0, 0])) / vxf
            src_vol = nibabel.Nifti1Image(scipy.ndimage.gaussian_filter(
                    src_vol.get_data(), fwhm2sigma(fwhmf)),
                                          src_vol.get_affine())

        # piramidal loop
        self.params_ = self.initial_params
        for samp in self.sep:
            # powell gradient-less local optimization
            self._log(
                ("Running powell gradient-less local optimization "
                 "(sampling=%smm)...") % samp)
            self.params_ = spm_powell(self.params_, self.xi, self.sc, ref_vol,
                                      src_vol, samp, self.cost_fun, self.fwhm)
            self._log("...done.\r\n")

        return self

    def transform(self,
                  src_vol,
                  output_dir=None,
                  prefix="Coreg",
                  ext=".nii.gz",
                  concat=False
                  ):
        """
        Applies estimated co-registration parameter to the input volume
        (src_vol).

        Parameters
        ----------
        src_vol: string (existing filename) or 3D nibabel image object
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
        output: dict
            output dict. items are:
            coregistered_source: nibabel image object or existing filename
            (if output_dir is specified)
            the coregistered source volume

        """

        output = {}

        if isinstance(src_vol, basestring):
            src_vol = nibabel.load(src_vol)
        else:
            assert is_niimg(src_vol)

        output['coregistered_source'] = nibabel.Nifti1Image(src_vol.get_data(),
                                                            scipy.linalg.lstsq(
                spm_matrix(self.params_),
                src_vol.get_affine())[0])

        if not output_dir is None:
            if isinstance(src_vol, basestring):
                basename = get_basenames(src_vol)
            else:
                basename = 'coregistered_source'

            save_vol(output['coregistered_source'], output_dir=output_dir,
                     basename=basename, ext=ext, prefix=prefix)

        return output
