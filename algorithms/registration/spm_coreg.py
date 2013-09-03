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
                                get_basenames
                                )
from affine_transformations import (spm_matrix,
                                    nibabel2spm_affine
                                    )
from kernel_smooth import fwhm2sigma

Flags = namedtuple('Flags', 'fwhm sep cost_fun tol params')
EPS = np.finfo(float).eps


def loaduint8(filename, log=None):
    """Load data from file indicated by V into array of unsigned bytes.

    """

    def _progress_bar(msg):
        if not log is None:
            log(msg)
        else:
            print(msg)

    if isinstance(filename, basestring):
        nii_img = nibabel.load(filename)
    elif is_niimg(filename):
        nii_img = filename
        filename = filename.get_filename()
        filename = filename if not filename is None else "UNNAMED NIFTI"
    else:
        raise TypeError("Unsupported input type: %s" % type(filename))

    vol = nii_img.get_data()

    if vol.ndim == 4:
        vol = vol[..., 0]

    _progress_bar("Loading %s..." % filename)

    def _spm_slice_vol(p):
        """Gets data fir pth slice (place) of volume vol

        """

        return vol[..., p].copy()

    def _accumarray(subs, N):
        """Computes the frequency of each index in subs, extended as
        and array of length N

        """

        subs = np.array(subs)

        ac = np.zeros(N)

        for j in set(subs):
            ac[j] = len(np.nonzero(subs == j)[0])

        return ac

    # if len(V.pinfo.shape) == 1:
    #     V.pinfo = V.pinfo.reshape((-1, 1))

    # if V.pinfo.shape[1] == 1 and V.pinfo[0] == 2:
    #     mx = 0xFF * V.pinfo[0] + V.pinfo[1]
    #     mn = V.pinfo[1]
    # else:
    mx = -np.inf
    mn = np.inf
    _progress_bar("\tComputing min/max of %s..." % filename)
    for p in xrange(vol.shape[2]):
        img = _spm_slice_vol(p)
        # mx = max(img.max() + _paccuracy(V, p), mx)
        mx = max(img.max(), mx)
        mn = min(img.min(), mn)

    # another pass to find a maximum that allows a few hot-spots in the data
    nh = 2048
    h = np.zeros(nh)
    _progress_bar("\t2nd pass max/min of %s..." % filename)
    for p in xrange(vol.shape[2]):
        img = _spm_slice_vol(p)
        img = img[np.isfinite(img)]
        img = np.round((img + ((mx - mn) / (nh - 1) - mn)
                        ) * ((nh - 1) / (mx - mn)))
        h = h + _accumarray(img - 1, nh)

    tmp = np.hstack((np.nonzero(np.cumsum(h) / np.sum(h) > .9999)[0], nh))
    mx = (mn * nh - mx + tmp[0] * (mx - mn)) / (nh - 1)

    # load data from file indicated by V into an array of unsigned bytes
    uint8_dat = np.ndarray(vol.shape, dtype='uint8')
    for p in xrange(vol.shape[2]):
        img = _spm_slice_vol(p)

        # add white-noise before rounding to reduce aliasing artefact
        # acc = _paccuracy(V, p)
        acc = 0
        r = 0 if acc == 0 else np.random.randn(*img.shape) * acc

        # pth slice
        uint8_dat[..., p] = np.uint8(np.maximum(np.minimum(np.round((
                            img + r - mn) * (255. / (mx - mn))), 255.), 0.))

    _progress_bar("...done.")

    # return the data
    return nibabel.Nifti1Image(uint8_dat, nii_img.get_affine())


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


def spm_conv_vol(vol, filtx, filty, filtz, xoff, yoff, zoff):
    output = scipy.ndimage.convolve1d(vol, filtx, axis=0)

    return output


def smooth_uint8(V, fwhm):
    """Convolves the volume V in memory (fwhm in voxels).

    """

    lim = np.ceil(2 * fwhm)

    x  = np.arange(-lim[0], lim[0] + 1)
    x = smoothing_kernel(fwhm[0], x)
    x  = x / np.sum(x)

    y  = np.arange(-lim[1], lim[1] + 1)
    y = smoothing_kernel(fwhm[1], y)
    y  = y / np.sum(y)

    z  = np.arange(-lim[2], lim[2] + 1)
    z = smoothing_kernel(fwhm[2], z)
    z  = z / np.sum(z)
    i  = (len(x) - 1) / 2
    j  = (len(y) - 1) / 2
    k  = (len(z) - 1) / 2

    return spm_conv_vol(V.astype('float'), x, y, z, -i, -j, -k)


def optfun(x, VG, VF, s=[1, 1, 1], cf='mi', fwhm=[7., 7.]):
    """
    Returns
    -------
    o

    """

    x = np.array(x)

    # voxel sizes
    vxg = np.sqrt(np.sum(VG.get_affine()[:3, :3] ** 2, axis=0))
    sg = s / vxg

    # create the joint histogram
    M = np.dot(scipy.linalg.lstsq(VF.get_affine(),
                                  spm_matrix(x))[0],
                                  VG.get_affine())
    H = spm_hist2py.hist2py(M, VG.get_data(), VF.get_data(), sg)

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

    H = H + EPS
    sh = np.sum(H)
    H = H / sh
    s1 = np.sum(H, axis=0).reshape((-1, H.shape[0]), order='F')
    s2 = np.sum(H, axis=1).reshape((H.shape[1], -1), order='F')
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
        raise NotImplementedError("Unsupported cd: %s" % cf)

    return o


def spm_powell(x0, xi, tolsc, *otherargs):
    """
    Run Powell optimization.

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
        ref_vol = loaduint8(ref_vol, log=self._log)
        ref_vol = nibabel.Nifti1Image(ref_vol.get_data(),
                                      nibabel2spm_affine(ref_vol.get_affine()))

        # load src_vol
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

    def transform(self, src_vol, output_dir=None, prefix="c", ext=".nii.gz",
                  concat=False):
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
        prefix: string, optional (default 'r')
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
        sd = fetch_nyu_rest(os.path.join(
                os.environ['HOME'], "CODE/datasets/nyu_rest"))

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
                "/mnt/3t/edohmato/ABIDE/%s_*/%s_*/scans" % (
                    institute, institute))):
            subject_id = os.path.basename(os.path.dirname(
                    os.path.dirname(scans)))
            func = os.path.join(scans, "rest/resources/NIfTI/files/rest.nii")
            anat = os.path.join(scans,
                                "anat/resources/NIfTI/files/mprage.nii")

            yield subject_id, func, anat

    def _run_demo(func, anat):
        # fit SPMCoreg object
        spmcoreg = SPMCoreg().fit(func, anat)

        # apply coreg
        VFk = spmcoreg.transform(anat)['coregistered_source']

        # QA
        plot_registration(anat, func, title="before coreg")
        plot_registration(VFk, func, title="after coreg")
        plt.show()

    # ABIDE demo
    for subject_id, func, anat in _abide_factory():
        print "%s +++%s+++\r\n" % ("\t" * 5, subject_id)
        _run_demo(func, anat)

    # spm auditory demo
    _run_demo(*_spm_auditory_factory())

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
