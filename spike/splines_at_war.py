"""
:Module: splines_at_war
:Synopsis: differential calculus for (discretely sampled) images, using
our splines
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import PIL
import sys


def compute_gradient_along_axis(im, axis, grid=None, mode='wrap'):
    """Approximates the gradient of an N-dimensional image by filtering
    it along the given axis, with the derivative of a spline along that
    axis.

    Parameters
    ----------
    im: array_like
        multi-dimensional array whose gradient along given axis is to be
        computed
    axis: int
        axis along which gradient is to be computed
    grid: array_like, optional (default None)
        grid of cooridinates to work on (object returned by --say-- np.mgrid)
    mode: string, optional (default "wrap")
        controls behaviour of the interpolators at the boundaries (
        see scipy.ndimage, scipy.signal, etc., doc for explanations)

    Returns
    -------
    array_like of same shape as grid[0]
        computed gradient along axis

    References
    ----------
    .. [1] Michael Unser, "A PERFECT FIT FOR SIGNAL/IMAGE PROCESSING"

    """

    # order of interpolating spline
    n = 3

    # filter with spline filter of order n - 1 along axis
    filtered_im = scipy.ndimage.spline_filter1d(im, order=n - 1, axis=axis)

    # create working grid
    if grid is None:

        grid = np.indices(im.shape)

    # warp filtered image by translating -.5 units along axis
    g = grid.copy()
    g[axis] += -.5
    left = scipy.ndimage.map_coordinates(filtered_im,
                                         [g[j].ravel()
                                          for j in xrange(len(im.shape))],
                                         order=1,  # linear splines will do
                                         mode=mode,
                                         prefilter=False)

    # warp filtered image by translating +.5 units along axis
    g = grid.copy()
    g[axis] += .5
    right = scipy.ndimage.map_coordinates(filtered_im,
                                          [g[j].ravel()
                                           for j in xrange(len(im.shape))],
                                          order=1,
                                          mode=mode,
                                          prefilter=False,)

    # rigt - left equals the gradient along given axis
    return (right - left).reshape(grid[0].shape)


# demo
if __name__ == '__main__':
    # load the image
    im_filename = "flower_frame_0060_LEVEL0.jpg"
    if len(sys.argv) > 1:
        im_filename = sys.argv[1]
    im = np.asarray(PIL.Image.open(im_filename))

    # sanitize the image (we only do grayscale)
    if len(im.shape) == 3:
        im = im[..., ..., 0]
    else:
        assert len(im.shape) == 2

    # compute grad along x axis
    gradx = compute_gradient_along_axis(im, 0)

    # compute grad along y axis
    grady = compute_gradient_along_axis(im, 1)

    plt.figure()
    plt.gray()

    ax = plt.subplot2grid((2, 1), (0, 0))
    ax.imshow(im)
    ax.set_title("pic")

    ax1 = plt.subplot2grid((2, 3), (1, 0))
    ax1.imshow(gradx)
    ax1.set_title('gradx')

    ax2 = plt.subplot2grid((2, 3), (1, 1))
    ax2.imshow(grady)
    ax2.set_title('grady')
    ax2.imshow(grady)

    ax3 = plt.subplot2grid((2, 3), (1, 2))
    ax3.imshow(gradx ** 2 + grady ** 2)
    ax3.set_title('laplacian')
    ax3.imshow(grady)

    plt.show()
