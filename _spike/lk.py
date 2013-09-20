"""
:module: lk
:Synopsis: Regularized Lucas-Kanade optical flow algorithm with
iterative refinement
:author:elvis[do]dohmatob[at]inria[dot]fr

"""

import numpy as np
import scipy.signal
import scipy.linalg
import scipy.ndimage
import matplotlib.pyplot as plt
import sys
import PIL


def _compute_spatial_simple_grad_kernel(ndim=1):
    """Computes ndim-dimensional gradient kernel using
    a robust diff technique

    Parameters
    ----------
    ndim: int (optional, default 1)
        dimensionality of gradient kernel to be computed

    Returns
    -------
    grad_kernel: array of shape ndim x 2^ndim (i.e ndim ndim-dimensional
    hypercubes)
       gradient kernel; each grad_kernel[q] when convoluted with an
       ndim-dimensional image, gives gradient of the image along the q
       axis

    """

    # 1D differential element
    dq = np.array([-1., 1.]) / 2.

    # replicate dq ndim times
    while len(dq.shape) != ndim:
        dq = np.array([dq, ] * ndim)

    # return gradient kernel
    return np.array([dq.swapaxes(ndim - 1 - q, -1)
                     for q in xrange(ndim)])


def _compute_spatial_gradient(im, grad_kernel=None,):
    """Applies a gradient kernel to an image

    Parameters
    ----------
    img: ndim-dimensional array
        image whose gradient is sought for
    grad_kernel: ndim x 2^ndim-dimensional array (optional, default None)
        kernel w.r.t. gradient will be computed. If grad_kernel is None,
        it will be calculated

    XXX grad_kernel is seperable, exploit this to speed up computations!

    """

    ndim = len(im.shape)

    # sanitize grad kernel
    if grad_kernel == None:
        grad_kernel = _compute_spatial_simple_grad_kernel(ndim)

    # convolve image with kernel, along each axis thus computing the former's
    # spatial gradient
    return [scipy.signal.convolve(im, grad_kernel[q])
            for q in xrange(ndim)]


def LucasKanade(im1, im2, window=9, n_levels=1, alpha=.001, iterations=1):
    """Estimates the velocity field for im2 -> im1 registration using
    regularized Lucas-Kanade optical flow algorithm with iterative refinement

    Parameters
    ----------
    im1: 2D array of size (height, width)
        reference image
    im2: 2D array of same shape as im1
        whose motion relative to im1 is tobe estimated
    window: int (optional, default 9)
        regular patches of size window x window will be used in formulating
        the weighted LS problem
    n_levels: int (optional, default 1)
        number of levels in image pyramid
    alpha: float (optional, default .001)
        regularization (alpha = 0 implies no regularization)
    iterations: int (optional, default 1)
        computation budget, for iterative refinement

    Returns
    -------
    u: 2D array of same shape as the input images
        velocity field along x axis
    v: 2D array of same shape as the input images
        velocity field along y axis

    """

    # radius of window
    hw = int(np.floor(window / 2))

    # gradient kernel
    ndim = len(im1.shape)
    grad_kernel = _compute_spatial_simple_grad_kernel(ndim)

    # temporal filter
    temporal_kernel = np.ones(grad_kernel[0].shape) / 2.

    for p in xrange(n_levels):
        # init
        if p == 0:
            # flow velocity field
            u = np.zeros(im1.shape)
            v = np.zeros(im1.shape)
        else:
            # zoom flow velocity field
            raise RuntimeError("Parallel execution not implemented!")

        # refinement loop
        for r in xrange(iterations):
            print "Iteration %i/%i..." % (r + 1, iterations)

            # loop on every pixel
            for i in xrange(hw, im1.shape[0] - hw):
                for j in xrange(hw, im1.shape[1] - hw):
                    print "\tWorking on patch (%s-%s, %s-%s)..." % (i - hw,
                                                                    i + hw,
                                                                    j - hw,
                                                                    j + hw)

                    # grap a patch from im1
                    patch1 = im1[i - hw:i + hw + 1, j - hw:j + hw + 1]

                    # grap corresponding patch from im2
                    patch2 = im2[i - hw:i + hw + 1, j - hw:j + hw + 1]

                    # warp patch2 unto im2's space using current
                    # motion estimates
                    # XXX do affine_transform instead, and globally!
                    patch2 = scipy.ndimage.shift(patch2, [u, v])

                    # compute spatial gradient of patch1
                    Dx_im1, Dy_im1 = _compute_spatial_gradient(patch1,
                                                               grad_kernel)

                    # compute spatial gradient of patch2
                    Dx_im2, Dy_im2 = _compute_spatial_gradient(patch2,
                                                               grad_kernel)

                    # compute spatial gradient of film [patch1, patch2]
                    # along x axis
                    Dx = (Dx_im1 + Dx_im2)[1:window - 1,
                                           1:window - 1].T / 2.

                    # compute spatial gradient of film [patch1, patch2]
                    # along y axis
                    Dy = (Dy_im1 + Dy_im2)[1:window - 1,
                                           1:window - 1].T / 2.

                    # compute temporal gradient of film [patch1, patch2]
                    Dt_1 = scipy.signal.convolve(patch1,
                                                 temporal_kernel)
                    Dt_2 = scipy.signal.convolve(patch2,
                                                  temporal_kernel)
                    Dt = (Dt_1 - Dt_2)[1:window - 1, 1:window - 1].T / 2.

                    # make coefficient matrix A
                    A = np.vstack((Dx.ravel(), Dy.ravel())).T

                    # compute G = A'A
                    G = np.dot(A.T, A)

                    # regularize G (to ensure solubility of LS problem)
                    G[0, 0] += alpha
                    G[1, 1] += alpha

                    # solve WLS problem for velocity V = (Vx_ij, Vy_ij)
                    # patch1 -> patch2 (by hand!)
                    V = np.dot(np.dot([[G[1, 1], -G[0, 1]], [-G[1, 0],
                                                              G[0, 0]]],
                                      A.T), -Dt.ravel()) / scipy.linalg.det(G)

                    # update velocity field around point p = (i, j)
                    u[i, j] += V[0]
                    v[i, j] += V[1]

    # resizing
    u = u[window - 1:u.shape[0] - window + 1,
          window - 1:u.shape[1] - window + 1]
    v = v[window - 1:v.shape[0] - window + 1,
          window - 1:v.shape[1] - window + 1]

    # return flow velocity field
    return u, v


if __name__ == '__main__':
    # data acquisition
    im1_filename = 'flower_frame_0060_LEVEL0.jpg'
    im2_filename = 'flower_frame_0061_LEVEL0.jpg'
    if len(sys.argv) > 2:
        print sys.argv
        im1_filename, im2_filename = sys.argv[1:3]

    im1 = np.asarray(PIL.Image.open(im1_filename))

    if len(im1.shape) == 3:
        im1 = im1[:, :, 0]
    im2 = np.asarray(PIL.Image.open(im2_filename))
    if len(im2.shape) == 3:
        im2 = im2[:, :, 0]

    # estimate motion
    u, v = LucasKanade(im1, im2, iterations=5)

    # plot velocity field
    plt.quiver(u, v, headwidth=1, scale_units='xy', angles='xy', scale=3,
               color='b')

    # misc
    plt.xlabel("width")
    plt.ylabel("height")
    plt.title(
        ("Velocity field for motion estimated by Lucas-Kanade optical "
         "flow algorithm"))

    # show plots
    plt.show()
