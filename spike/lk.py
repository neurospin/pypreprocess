"""
:module: lk
:Synopsis: Regularized Lucas-Kanade optical flow algorithm with
iterative refinement
:author:elvis[do]dohmatob[at]inria[dot]fr

"""

import numpy as np
import scipy.signal
import scipy.linalg
import matplotlib.pyplot as plt


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
        u = np.round(u)
        v = np.round(v)
        for r in xrange(iterations):
            print "Iteration %i/%i..." % (r + 1, iterations)

            # loop on every pixel
            for i in xrange(hw, im1.shape[0] - hw):
                for j in xrange(hw, im1.shape[1] - hw):
                    print "\tWorking on patch (%s-%s, %s-%s)..." % (i - hw,
                                                                    i + hw,
                                                                    j - hw,
                                                                    j + hw)

                    patch1 = im1[i - hw:i + hw + 1, j - hw:j + hw + 1]

                    # move patch: resample grid for im2
                    lr = i - hw + v[i, j] + 1
                    hr = i + hw + v[i, j] + 1
                    lc = j - hw + u[i, j] + 1
                    hc = j + hw + u[i, j] + 1

                    if lr < 1 or hr > im1.shape[0] or lc < 1 or hc >\
                            im1.shape[1]:
                        raise Warning("Regularized LS not implemented!")
                    else:
                        # resample patch on im2 according current motion
                        # estimates
                        patch2 = im2[lr - 1:hr, lc - 1:hc]

                        # compute spatial gra0dient along x axis using Sobel
                        # filter
                        Dx_im1 = scipy.signal.convolve(patch1,
                                                        .25 * np.array(
                                [[-1, 1], [-1, 1]]))
                        Dx_im2 = scipy.signal.convolve(patch2,
                                                        .25 * np.array(
                                [[-1, 1], [-1, 1]]))
                        Dx = (Dx_im1 + Dx_im2)[1:window - 1, 1:window - 1].T

                        # compute spatial gradient along y axis using Sobel
                        # filter
                        Dy_im1 = scipy.signal.convolve(patch1,
                                                        .25 * np.array(
                                [[-1, -1], [1, 1]]))
                        Dy_im2 = scipy.signal.convolve(patch2,
                                                        .25 * np.array(
                                [[-1, -1], [1, 1]]))
                        Dy = (Dy_im1 + Dy_im2)[1:window - 1, 1:window - 1].T

                        # compute temporal gradient
                        Dt_1 = scipy.signal.convolve(patch1,
                                                      .25 * np.ones((2, 2)))
                        Dt_2 = scipy.signal.convolve(patch2,
                                                      .25 * np.ones((2, 2)))
                        Dt = (Dt_1 - Dt_2)[1:window - 1, 1:window - 1].T

                        # make (rank-deficient) coefficient matrix A
                        A = np.vstack((Dx.ravel(), Dy.ravel())).T

                        # compute G = A'A
                        G = np.dot(A.T, A)

                        # dope G
                        G[0, 0] += alpha
                        G[1, 1] += alpha

                        # solve WLS problem for velocity V = (Vx_ij, Vy_ij)
                        # patch1 -> patch2
                        V = scipy.linalg.lstsq(G, -np.dot(A.T, Dt.ravel()))[0]

                        # update velocity field around point p = (i, j)
                        u[i, j] += V[0]
                        v[i, j] += V[1]

    # return flow velocity field
    return u, v


if __name__ == '__main__':
    # data acquisition
    im1 = plt.imread('flower_frame_0060_LEVEL0.jpg')
    im2 = plt.imread('flower_frame_0061_LEVEL0.jpg')

    # estimate motion
    u, v = LucasKanade(im1, im2)

    # plot velocity field
    plt.quiver(u, v)

    # show plots
    plt.show()
