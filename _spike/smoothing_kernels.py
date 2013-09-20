"""
:Author: elvis[dot]dohmatob[at]inria[dot]fr

"""

import numpy as np


def solve_wlstsq(K, X, Y, verbose=1):
    """Solves a kernel-weighted Least-Squares problem with nonnegative
    weights, and 2 regressors (one covariate and the intercept / constant /
    0th regressor). All algebraic operations have been extensively
    "vectorized".

    References
    ----------
    See 'The Elements of Statistical Learning ', 2nd edition page 195 for
    a statement of the problem related to 'local linear regression'

    Parameters
    ----------
    K: array
        2D array of shape (n_user_times, n_scans) of kernel weights. Each
        row of K defines a kernel around the corresponding user time

    X: array
       Feature vector, same size as `K`, one row per user time

    Y: array
       Response vector, same size as `K`, one row per user time

    Returns
    -------
    alpha_opt: array
        optimimal intercepts, 1D array of size n_user_times

    beta_opt: float
        optiminal regressor coefficients, 1D array of size n_user_times

    """

    # sanitize dimensions
    if len(K.shape) == 1:
        K = K.reshape((1, len(K)))
        try:
            X = X.reshape((1, len(K)))
            Y = Y.reshape((1, len(K)))
        except ValueError:
            raise ValueError(
                "K, X, and X must be of same shape (n_user_times, n_scans)")

    n_user_times, n_scans = K.shape
    assert X.shape == Y.shape == (n_user_times, n_scans)

    # compute aux variables
    b = np.dot(K, X.T)
    a = K.sum(axis=1)
    a = a.reshape((n_user_times, 1))
    c = np.dot(K, Y.T)
    d = np.dot(K, (X * X).T)
    e = np.dot(K, (X * Y).T)

    # compute optimal beta
    beta_opt = np.diag((a * e - b * c) / (a * d - b * b)).reshape(
        (n_user_times, 1))

    # kill all NaNs in beta_opt
    evil_places = np.nonzero(np.isnan(beta_opt) | np.isinf(beta_opt))[0]
    beta_opt[evil_places, 0] = 0.

    # compute optima alpha
    alpha_opt = np.diag(np.dot(K, (Y - beta_opt * X).T) / a)

    # kill all alphas corresponding to killed betas; this way we fit an
    # infinitesimally short line segment, i.e a point
    alpha_opt[evil_places] = 0.

    # sanitize beta's shape
    beta_opt = beta_opt.ravel()

    # brag about how good we're
    if verbose:
        _ones = np.ones(n_scans)
        print "[+] solve_wlstsq(...) results:"
        for j in xrange(n_user_times):
            z = (Y[j, :] - alpha_opt[j] * _ones - beta_opt[j] * X[j, :])
            q = np.dot(K[j, :], z * z)
            print "\t[user time index %i] argmin = (%f, %f), min = %f" % (
                j,
                alpha_opt[j],
                beta_opt[j], q)

    # return to caller
    return alpha_opt, beta_opt


def epanechnikov_kernel(t, L):
    """The Epnaechnikov kernel (compact support, suitable for kernels based on
    k-nearest neighborhoods.

    """

    if L > 0:
        return 0. if np.abs(t) > L else .75 * (1 - (t / L) ** 2)
    else:
        return 1. if t == 0 else 0.


def apply_llr_kernel(x_0, alpha_opt, beta_opt):
    y_0 = alpha_opt + beta_opt * x_0

    return y_0


def llreg(x_0, X, Y, k=2, jobtype="estimate", voxel_slice_index=None,
          verbose=1):
    """Linear Local Regression

    Parameters
    ----------
    x_0: float or array of floats
        point(s) at which we wish to interpolate/extrapolate

    X: array
       realizations of predictor variable

    Y: array, same shape as X
       realizations of E(Y|X=x)

    k: int (optional, default 5)
       we'll use k-nearest neighborhoods to define kernel width

    """

    # sanity
    assert len(X.shape) == 1

    if not hasattr(x_0, '__len__'):
        x_0 = np.array([x_0])

    n_user_times = len(x_0)

    # compute kernel width based on k-nearest neighborhoods
    kernel_width = np.array(
        [sorted(np.abs(a - X))[k] for a in x_0])
    kernel_width[0] = 1

    # compute kernel
    kernel = np.array([
            np.vectorize(epanechnikov_kernel)(x_0[j] - X, kernel_width[j])
            for j in xrange(n_user_times)])
    print kernel

    # solve kernel-weighted LS problem at x_0 using my solve_wlstsq algo
    X = np.array([X for _ in xrange(n_user_times)])
    Y = np.array([Y for _ in xrange(n_user_times)])

    if not voxel_slice_index is None:
        print ("[+] Estimating LLR (Local Linear Regression) kernel for voxel"
               " %i of slice %i") % (
            voxel_slice_index[0], voxel_slice_index[1])
    alpha_opt, beta_opt = solve_wlstsq(kernel, X, Y, verbose=verbose)

    if jobtype == "estimate":
        return alpha_opt, beta_opt

    # predict the value at x_0
    y_0 = apply_llr_kernel(x_0, alpha_opt, beta_opt)

    return y_0


if __name__ == '__main__':
    n_scans = 100

    # K = np.array([np.random.random(n_scans) for j in xrange(n_user_times)])
    # X = np.random.randn(n_scans * n_user_times).reshape((n_user_times,
    #                                                      n_scans))
    # Y = np.random.randn(n_scans * n_user_times).reshape((n_user_times,
    #                                                      n_scans))

    # sample the traing domain
    X = np.linspace(-10, 10, n_scans)
    Y = X ** 3 - X

    # add some wicked white noise
    Y += 100 * np.random.rand(len(X))

    # define the targe points (the points for which you want to predict the
    # missing values)
    x_0 = X + .2  # np.array([.5, 1.5, 2.5, 3.5, 4.5])

    # do ya thing
    y_0 = llreg(x_0, X, Y, jobtype='estwrite')

    # visualize the results
    import pylab as pl
    pl.plot(X, Y)
    pl.hold('on')
    pl.plot(x_0, y_0, 'o')
    pl.title("Local Polynomial Regression")
    pl.legend(("Training sample", "Predicting values"))
    pl.show()
