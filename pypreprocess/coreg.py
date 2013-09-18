"""
:Author: DOHMATOB Elvis Dopgima

"""

import sys
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

# pypreprocess dir
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

from coreutils.io_utils import (is_niimg,
                                save_vol,
                                get_basenames,
                                load_specific_vol
                                )
from affine_transformations import (spm_matrix,
                                    nibabel2spm_affine
                                    )
from kernel_smooth import fwhm2sigma

# flags for fitting coregistration model
Flags = namedtuple('Flags', 'fwhm sep cost_fun tol params')

# 'texture' of floats in machine presicion
EPS = np.finfo(float).eps


def loaduint8(img, log=None):
    """Load data from file indicated by V into array of unsigned bytes.

    Parameters
    ----------
    img: string, `np.ndarray`, or niimg
        image to be loaded

    Returns
    -------
    uint8_data: `np.ndarray`, if input was ndarray; `nibabel.NiftiImage1' else
        the loaded image (dtype='uint8')

    """

    def _progress_bar(msg):
        """
        Progress bar.

        """

        if not log is None:
            log(msg)
        else:
            print(msg)

    _progress_bar("Loading %s..." % img)

    # load volume into memory
    if isinstance(img, np.ndarray) or isinstance(img, list):
        vol = np.array(img)
    elif isinstance(img, basestring):
        img = nibabel.load(img)
        vol = img.get_data()
    elif is_niimg(img):
        vol = img.get_data()
    else:
        raise TypeError("Unsupported input type: %s" % type(img))

    if vol.ndim == 4:
        vol = vol[..., 0]

    assert vol.ndim == 3

    def _spm_slice_vol(p):
        """
        Gets data pth slice of vol.

        """

        return vol[..., p].copy()

    # min/max
    mx = -np.inf
    mn = np.inf
    _progress_bar("\tComputing min/max...")
    for p in xrange(vol.shape[2]):
        _img = _spm_slice_vol(p)
        mx = max(_img.max(), mx)
        mn = min(_img.min(), mn)

    # load data from file indicated by V into an array of unsigned bytes
    uint8_dat = np.ndarray(vol.shape, dtype='uint8')
    for p in xrange(vol.shape[2]):
        _img = _spm_slice_vol(p)

        # pth slice
        uint8_dat[..., p] = np.uint8(np.maximum(np.minimum(np.round((
                            _img - mn) * (255. / (mx - mn))), 255.), 0.))

    _progress_bar("...done.")

    # return the data
    if isinstance(img, basestring) or is_niimg(img):
        return nibabel.Nifti1Image(uint8_dat, img.get_affine())
    else:
        return uint8_dat


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

if __name__ == '__main__':
    import glob
    import matplotlib.pyplot as plt
    from external.nilearn.datasets import (fetch_spm_auditory_data,
                                           fetch_nyu_rest,
                                           fetch_spm_multimodal_fmri_data
                                           )
    from algorithms.registration.spm_realign import _apply_realignment_to_vol
    from algorithms.registration.affine_transformations import spm_matrix
    from reporting.check_preprocessing import plot_registration
    from coreutils.io_utils import delete_orientation

    def _nyu_rest_factory():
        sd = fetch_nyu_rest(data_dir=os.path.join(
                os.environ['HOME'], "CODE/datasets/nyu_rest"), sessions=[1],
                            n_subjects=7)

        for j in xrange(len(sd.func)):
            output_dir = os.path.join("/tmp", os.path.basename(
                    os.path.dirname(os.path.dirname(sd.func[j]))))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            yield delete_orientation(sd.func[j], output_dir
                                     ), delete_orientation(sd.anat_skull[j],
                                                           output_dir)

    def _spm_auditory_factory():
        sd = fetch_spm_auditory_data(os.path.join(
                os.environ['HOME'], "CODE/datasets/spm_auditory"))

        return sd.func[0], sd.anat

    def _abide_factory(institute="KKI"):
        for scans in sorted(glob.glob(
                "/home/elvis/CODE/datasets/ABIDE/%s_*/%s_*/scans" % (
                    institute, institute))):
            subject_id = os.path.basename(os.path.dirname(
                    os.path.dirname(scans)))
            func = os.path.join(scans, "rest/resources/NIfTI/files/rest.nii")
            anat = os.path.join(scans,
                                "anat/resources/NIfTI/files/mprage.nii")

            yield subject_id, func, anat

    def _run_demo(func, anat):
        # fit SPMCoreg object
        spmcoreg = SPMCoreg().fit(anat, func)

        # apply coreg
        VFk = load_specific_vol(spmcoreg.transform(
                func)['coregistered_source'], 0)[0]

        # QA
        plot_registration(anat, VFk, title="before coreg")
        plot_registration(VFk, anat, title="after coreg")
        plt.show()

    # ABIDE demo
    for subject_id, func, anat in _abide_factory():
        print "%s +++%s+++\r\n" % ("\t" * 5, subject_id)
        _run_demo(func, anat)

    # # spm auditory demo
    # _run_demo(*_spm_auditory_factory())

    # abide rest demo
    for func, anat in _abide_factory():
        _run_demo(func, anat)

    # VFk = _apply_realignment_to_vol(VFk, q0)
    # print q0

    # shape = (2, 2)
    # p = np.zeros(6)
    # spmcoreg = SPMCoreg()
    # for i, j in np.ndindex(shape):
    #     p[:3] = np.random.randn(3) * (i + j) * 2
    #     p[3:] = np.random.randn(3) * .1

    #     x = _apply_realignment_to_vol(sd.func[0], p)
    #     q = spmcoreg.fit(sd.func[0], x).params_

    #     ax = plt.subplot2grid(shape, (i, j))
    #     ax.plot(np.transpose([p, -q]), 's-')

    plt.show()
