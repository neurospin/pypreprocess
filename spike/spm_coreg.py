import numpy as np
import scipy.ndimage
import scipy.special
import scipy.io
import nibabel

EPS = 0.


def loaduint8(V):
    """Load data from file indicated by V into array of unsigned bytes.

    """

    vol = nibabel.load(V.fname).get_data()[..., 0]

    def _spm_slice_vol(V, p):
        """Gets data fir pth slice (place) of volume V

        """

        return vol[..., p]

    def _paccuracy(V, p):
        """Computes the accuracy limit of rounding intensities into uint8 type

        """

        if isinstance(V.dt[0], int):
            acc = 0
        else:
            if V.pinfo.shape[1] == 1:
                acc = np.abs(V.pinfo[0, 0])
            else:
                acc = np.abs(V.pinfo[0, p])

        return acc

    def _accumarray(subs, N):
        """Computes the frequency of each index in subs, extended as
        and array of length N

        """

        subs = np.array(subs)

        ac = np.zeros(N)

        for j in set(subs):
            ac[j] = len(np.nonzero(subs == j)[0])

        return ac

    if len(V.pinfo.shape) == 1:
        V.pinfo = V.pinfo.reshape((-1, 1))

    if V.pinfo.shape[1] == 1 and V.pinfo[0] == 2:
        mx = 0xFF * V.pinfo[0] + V.pinfo[1]
        mn = V.pinfo[1]
    else:
        mx = -np.inf
        mn = np.inf
    for p in xrange(V.dim[2]):
        img = _spm_slice_vol(V, p)
        mx = max(img.max() + _paccuracy(V, p), mx)
        mn = min(img.min(), mn)

    # another pass to find a maximum that allows a few hot-spots in the data
    nh = 2048
    h = np.zeros(nh)
    for p in xrange(V.dim[2]):
        img = _spm_slice_vol(V, p)
        img = img[np.isfinite(img)]
        img = np.round((img + ((mx - mn) / (nh - 1) - mn)
                        ) * ((nh - 1) / (mx - mn)))
        h = h + _accumarray(img - 1, nh)

    tmp = np.hstack((np.nonzero(np.cumsum(h) / np.sum(h) > .9999)[0], nh))
    mx = (mn * nh - mx + tmp[0] * (mx - mn)) / (nh - 1)

    # load data from file indicated by V into an array of unsigned bytes
    uint8_dat = np.ndarray(V.dim, dtype='uint8')
    for p in xrange(V.dim[2]):
        img = _spm_slice_vol(V, p)

        # add white-noise before rounding to reduce aliasing artefact
        acc = _paccuracy(V, p)
        r = 0 if acc == 0 else np.random.randn(*img.shape) * acc

        # pth slice
        uint8_dat[..., p] = np.uint8(np.maximum(np.minimum(np.round((
                            img + r - mn) * (0xFF / (mx - mn))), 0xFF), 0x00))

    # return the data
    return uint8_dat


def fwhm2sigma(fwhm):
    """Convert a FWHM value to sigma in a Gaussian kernel.

    Parameters
    ----------
    fwhm: array-like
       FWHM value or values

    Returns
    -------
    sigma: array or float
       sigma values corresponding to `fwhm` values

    Examples
    --------
    >>> sigma = fwhm2sigma(6)
    >>> sigmae = fwhm2sigma([6, 7, 8])
    >>> sigma == sigmae[0]
    True

    """

    fwhm = np.asarray(fwhm)

    return fwhm / np.sqrt(8 * np.log(2))


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
    pass


def smooth_uint8(V, fwhm):
    """Convolves the volume V in memory (fwhm in voxels).

    """

    lim = np.ceil(2 * fwhm)

    x  = np.arange(-lim[0], lim[0] + 1)
    x = smoothing_kernel(fwhm(1), x)
    x  = x / np.sum(x)

    y  = np.arange(-lim[1], lim[1] + 1)
    y = smoothing_kernel(fwhm(2), y)
    y  = y / np.sum(y)

    z  = np.arange(-lim[2], lim[2] + 1)
    z = smoothing_kernel(fwhm(3), z)
    z  = z / np.sum(z)
    i  = (len(x) - 1) / 2
    j  = (len(y) - 1) / 2
    k  = (len(z) - 1) / 2

    spm_conv_vol(V.uint8, V.uint8, x, y, z, [-i, -j, -k])

if __name__ == '__main__':
    from collections import namedtuple
    Volume = namedtuple('Volume', 'fname dt dim pinfo')
    V = Volume(
        fname="/home/elvis/CODE/datasets/spm_auditory/fM00223/fM00223_004.img",
        dt=[1, 1],
        dim=[64, 64, 64],
        pinfo=np.array([[4, 1]])
        )
    # V = scipy.io.loadmat('/tmp/V.mat', squeeze_me=True,
    #                      struct_as_record=False)['V']

    uV = loaduint8(V)
