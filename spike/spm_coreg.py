import numpy as np
import scipy.ndimage
import scipy.special
import scipy.io
import nibabel

EPS = 0.


# def _paccuracy(V, p):
#     """Computes the accuracy limit of rounding intensities into uint8 type

#     """

#     if isinstance(V.dt[0], int):
#         acc = 0
#     else:
#         if V.pinfo.shape[1] == 1:
#             acc = np.abs(V.pinfo[0, 0])
#         else:
#             acc = np.abs(V.pinfo[0, p])

#     return acc


def loaduint8(filename):
    """Load data from file indicated by V into array of unsigned bytes.

    """

    if isinstance(filename, basestring):
        nii_img = nibabel.load(filename)
    else:
 # isinstance(filename, nibabel.Nifti1Image) or \
 #            isinstance(filename, nibabel.Nifti1Pair):
        nii_img = filename
        filename = filename.get_filename()
    # else:
    #     raise TypeError("Unsupported input type: %s" % type(filename))

    vol = nii_img.get_data()

    if vol.ndim == 4:
        vol = vol[..., 0]

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

    def _progress_bar(msg):
        print(msg)

    # if len(V.pinfo.shape) == 1:
    #     V.pinfo = V.pinfo.reshape((-1, 1))

    # if V.pinfo.shape[1] == 1 and V.pinfo[0] == 2:
    #     mx = 0xFF * V.pinfo[0] + V.pinfo[1]
    #     mn = V.pinfo[1]
    # else:
    mx = -np.inf
    mn = np.inf
    _progress_bar("Computing min/max of %s..." % filename)
    for p in xrange(vol.shape[2]):
        img = _spm_slice_vol(p)
        # mx = max(img.max() + _paccuracy(V, p), mx)
        mx = max(img.max(), mx)
        mn = min(img.min(), mn)

    # another pass to find a maximum that allows a few hot-spots in the data
    nh = 2048
    h = np.zeros(nh)
    _progress_bar("2nd pass max/min of %s..." % filename)
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
    print "Loading %s..." % filename
    for p in xrange(vol.shape[2]):
        img = _spm_slice_vol(p)

        # add white-noise before rounding to reduce aliasing artefact
        # acc = _paccuracy(V, p)
        acc = 0
        r = 0 if acc == 0 else np.random.randn(*img.shape) * acc

        # pth slice
        uint8_dat[..., p] = np.uint8(np.maximum(np.minimum(np.round((
                            img + r - mn) * (0xFF / (mx - mn))), 0xFF), 0x00))

    # return the data
    return nibabel.Nifti1Image(uint8_dat, nii_img.get_affine())


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

    spm_conv_vol(V.uint8, x, y, z, -i, -j, -k)


def _tpvd_interp(f, fshape, x, y, z):
    """Performs "trilinear partial volume distribution" interpolation of a
    gray-scale 3D image f, at a voxel (x, y, z).

    Parameters
    ----------
    f: array_like of floats
        gray-scale image to be interpolated
    x: float or array_like of floats
        ...

    """

    ix = np.floor(x)
    dx1 = x - ix
    dx2 = 1.0 - dx1

    iy = np.floor(y)
    dy1 = y - iy
    dy2 = 1.0 - dy1

    iz = np.floor(z)
    dz1 = z - iz
    dz2 = 1.0 - dz1

    ff = f[ix - 1 + fshape[0] * (iy - 1 + fshape[1] * (iz - 1)):]
    k222 = ff[0]
    k122 = ff[1]
    k212 = ff[fshape[0]]
    k112 = ff[fshape[0] + 1]
    ff = ff[fshape[0] * fshape[1]:]
    k221 = ff[0]
    k121 = ff[1]
    k211 = ff[fshape[0]]
    k111 = ff[fshape[0] + 1]

    vf = (((k222 * dx2 + k122 * dx1) * dy2  +\
               (k212 * dx2 + k112 * dx1) * dy1)) * dz2 +\
               (((k221 * dx2 + k121 * dx1) * dy2 +\
                     (k211 * dx2 + k111 * dx1) * dy1)) * dz1

    return vf


def _joint_histogram(M, g, f, gshape, fshape, s=[1, 1, 1]):
    """
    Computes the joint histogram of g and f[warp(f, M)],
    where M is an affine transformation, and g and f are
    3-dimensional images (scalars defined on the vertices of polytopes)
    of possible different shapes (i.e different resolutions).

    Parameters
    ----------
    f: array_like
        3-dimensional image
    g: array_like
        3-dimensional image
    M: array_like of shape (4, 4)
        affine transformation with which f will be warped before
        computing the histogram

    Returns
    -------
    H: joint histogram, structure from numpy.histogram2d back-end.

    """

    ran = np.array([0.656619, 0.891183, 0.488144, 0.992646, 0.373326, 0.531378,
                    0.181316, 0.501944, 0.422195, 0.660427, 0.673653, 0.95733,
                    0.191866, 0.111216, 0.565054, 0.969166, 0.0237439,
                    0.870216, 0.0268766, 0.519529, 0.192291, 0.715689,
                    0.250673, 0.933865, 0.137189, 0.521622, 0.895202,
                    0.942387, 0.335083, 0.437364, 0.471156, 0.14931, 0.135864,
                    0.532498, 0.725789, 0.398703, 0.358419, 0.285279, 0.868635,
                    0.626413, 0.241172, 0.978082, 0.640501, 0.229849, 0.681335,
                    0.665823, 0.134718, 0.0224933, 0.262199, 0.116515,
                    0.0693182, 0.85293, 0.180331, 0.0324186, 0.733926,
                    0.536517, 0.27603, 0.368458, 0.0128863, 0.889206, 0.866021,
                    0.254247, 0.569481, 0.159265, 0.594364, 0.3311, 0.658613,
                    0.863634, 0.567623, 0.980481, 0.791832, 0.152594,
                    0.833027, 0.191863, 0.638987, 0.669, 0.772088, 0.379818,
                    0.441585, 0.48306, 0.608106, 0.175996, 0.00202556,
                    0.790224, 0.513609, 0.213229, 0.10345, 0.157337, 0.407515,
                    0.407757, 0.0526927, 0.941815, 0.149972, 0.384374,
                    0.311059, 0.168534, 0.896648
                    ])

    # compute the joint histogram H for g and f
    H = np.zeros(256 * 256)  # assuming 8-bit grayscale
    iran = 0
    z = 1.
    while z < gshape[2] - s[2]:
        y = 1.
        while y < gshape[1] - s[1]:
            x = 1.
            while x < gshape[0] - s[0]:
                # print (x, y, z)
                iran = (iran + 1) % 97
                rx  = x + ran[iran] * s[0]
                iran = (iran + 1) % 97
                ry  = y + ran[iran] * s[1]
                iran = (iran + 1) % 97
                rz  = z + ran[iran] * s[2]

                # map voxel (rx, ry, rz) under the affine transformation
                xp, yp, zp, _ = np.dot(M, [rx, ry, rz, 1.])

                # hereunder, if we haven't fallen out of the FOV, we update the
                # bin to which the joint intensity pair (g[rx, ry, rz],
                # f[xp, yp, zp]) belongs.
                if ((zp >= 1.) and (zp < fshape[2]) and (yp >= 1.) and
                    (yp < fshape[1]) and (xp >= 1.) and (xp < fshape[0])):

                    print (rx, ry, rz), '->', (xp, yp, zp)

                    # interpolate f at (rx, ry, rz)
                    vf  = _tpvd_interp(f, fshape, xp, yp, zp)
                    ivf = np.floor(vf).astype('int')

                    # interpolate g at voxel (xp, yp, zp)
                    ivg = np.floor(_tpvd_interp(g, gshape, rx, ry, rz) + 0.5
                                   ).astype('int')

                    # update corresponding bin
                    H[(ivf + ivg * 256)] += (1 - (vf - ivf))

                    # handle special boundary condition
                    if (ivf < 255):
                        H[(ivf + 1 + ivg * 256)] += (vf - ivf)

                # update x
                x += s[0]

            # update y
            y += s[1]

        # update z
        z += s[2]

    # return 256 x 256 (fortran-order!) joint histogram
    return H.reshape((256, 256), order='F')


def optfunc(x, VG, VF, s=[1, 1, 1], cf='mi', fwhm=[7, 7]):
    """The cost function minimized by the coregistration algorithm (Powell)

    """

    # voxel sizes
    vxg = np.sqrt(np.sum(VG.get_affine()[:3, :3] ** 2, axis=0))
    sg = s / vxg

    # create the joint histogram
    pass

if __name__ == '__main__':
    # from collections import namedtuple
    # Volume = namedtuple('Volume', 'fname dt dim pinfo')
    # V = Volume(
    #     fname=("/home/edohmato/CODE/datasets/spm_auditory"
    #            "/fM00223/fM00223_004.img"),
    #     dt=[1, 1],
    #     dim=[64, 64, 64],
    #     pinfo=np.array([[4, 1]])
    #     )
    # # V = scipy.io.loadmat('/tmp/V.mat', squeeze_me=True,
    # #                      struct_as_record=False)['V']

    uV = loaduint8(("/home/edohmato/CODE/datasets/spm_auditory"
                    "/fM00223/fM00223_004.img"))
