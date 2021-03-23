"""
:Module: check_preprocessing
:Synopsis: module for generating post-preproc plots (registration,
segmentation, etc.)
:Author: bertrand thirion, dohmatob elvis dopgima

"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel
from nilearn.plotting import plot_img
from nilearn.image import reorder_img, mean_img
from ..io_utils import load_vols
EPS = np.finfo(float).eps

import io
import base64
import urllib.parse

def _plot_to_svg(fig, dpi=300):
    """ 
    Converts matplotlib figure instance to an SVG url
    that can be loaded in a browser.

    Parameters
    ----------
    fig: `matplotlib.figure.Figure` instance 
        consisting of the plot to be converted to SVG
        url and then enbedded in an HTML report.

    dpi: float, optional (default 300)
        Dots per inch. Resolution of the SVG plot generated
    """
    with io.BytesIO() as io_buffer:
        fig.tight_layout(pad=0.4)
        fig.savefig(
            io_buffer, format="svg", facecolor="white",
            edgecolor="white", dpi=dpi)
        return urllib.parse.quote(io_buffer.getvalue().decode("utf-8"))


def plot_spm_motion_parameters(parameter_file, lengths,
                            title=None, output_filename=None,
                            close=False, report_path=None):
    """ Plot motion parameters obtained with SPM software

    Parameters
    ----------
    parameter_file: string
        path of file containing the motion parameters
    subject_id: string (optional)
        subject id
    titile: string (optional)
        title to attribute to plotted figure
    output_filename: string
        output filename for storing the plotted figure

    """
    # load parameters
    motion = np.loadtxt(parameter_file) if isinstance(
        parameter_file, str) else parameter_file[..., :6]

    motion[:, 3:] *= (180. / np.pi)

    # do plotting
    plt.figure()
    plt.plot(motion)

    aux = 0.
    for l in lengths[:-1]:
        plt.axvline(aux + l, linestyle="--", c="k")
        aux += l

    if not title is None:
        plt.title(title)
    plt.legend(('TransX', 'TransY', 'TransZ', 'RotX', 'RotY', 'RotZ'),
               loc="upper left", ncol=2)
    plt.xlabel('time(scans)')
    plt.ylabel('Estimated motion (mm/degrees)')

    if report_path not in [False, None]:
        fig = plt.gcf()
        svg_plot = _plot_to_svg(fig)
    else: 
        svg_plot = None

    if not output_filename is None:
        plt.savefig(output_filename, bbox_inches="tight", dpi=200)
        if close:
            plt.close()

    return svg_plot


def compute_cv(data, mask_array=None):
    if mask_array is not None:
        cv = .0 * mask_array
        cv[mask_array > 0] = data[mask_array > 0].std(-1) /\
            (data[mask_array > 0].mean(-1) + EPS)
    else:
        cv = data.std(-1) / (data.mean(-1) + EPS)

    return cv


def plot_registration(reference_img, coregistered_img,
                      title="untitled coregistration!",
                      cut_coords=None,
                      display_mode='ortho',
                      cmap=None, close=False,
                      output_filename=None,
                      report_path=None):
    """Plots a coregistered source as bg/contrast for the reference image

    Parameters
    ----------
    reference_img: string
        path to reference (background) image

    coregistered_img: string
        path to other image (to be compared with reference)

    display_mode: string (optional, defaults to 'ortho')
        display_mode param

    cmap: matplotlib colormap object (optional, defaults to spectral)
        colormap to user for plots

    output_filename: string (optional)
        path where plot will be stored

    """
    # sanity
    if cmap is None:
        cmap = plt.cm.gray  # registration QA always gray cmap!

    reference_img = mean_img(reference_img)
    coregistered_img = mean_img(coregistered_img)

    if cut_coords is None:
        cut_coords = (-10, -28, 17)

    if display_mode in ['x', 'y', 'z']:
        cut_coords = (cut_coords['xyz'.index(display_mode)],)

    # XXX nilearn complains about rotations in affine, etc.
    coregistered_img = reorder_img(coregistered_img, resample="continuous")

    _slicer = plot_img(coregistered_img, cmap=cmap, cut_coords=cut_coords,
                       display_mode=display_mode, black_bg=True)

    # XXX nilearn complains about rotations in affine, etc.
    reference_img = reorder_img(reference_img, resample="continuous")

    _slicer.add_edges(reference_img)
    # misc
    _slicer.title(title, size=12, color='w', alpha=0)

    if report_path not in [False, None]:
        fig = plt.gcf()
        svg_plot = _plot_to_svg(fig)
    else:
        svg_plot = None

    if not output_filename is None:
        try:
            plt.savefig(output_filename, dpi=200, bbox_inches='tight',
                        facecolor="k", edgecolor="k")
            if close:
                plt.close()
        except AttributeError:
            # XXX TODO: handle this case!!
            pass

    return svg_plot

def plot_segmentation(
        img, gm_filename, wm_filename=None, csf_filename=None,
        output_filename=None, cut_coords=None, display_mode='ortho',
        cmap=None, title='GM + WM + CSF segmentation', close=False,
        report_path=None):
    """
    Plot a contour mapping of the GM, WM, and CSF of a subject's anatomical.

    Parameters
    ----------
    img_filename: string or image object
                  path of file containing image data, or image object simply

    gm_filename: string
                 path of file containing Grey Matter template

    wm_filename: string (optional)
                 path of file containing White Matter template

    csf_filename: string (optional)
                 path of file containing Cerebro-Spinal Fluid template

    """
    # misc
    if cmap is None:
        cmap = plt.cm.gray
    if cut_coords is None:
        cut_coords = (-10, -28, 17)
    if display_mode in ['x', 'y', 'z']:
        cut_coords = (cut_coords['xyz'.index(display_mode)],)

    # plot img
    img = mean_img(img)
    img = reorder_img(img, resample="continuous")
    _slicer = plot_img(img, cut_coords=cut_coords, display_mode=display_mode,
                       cmap=cmap, black_bg=True)

    # add TPM contours
    gm = nibabel.load(gm_filename)
    _slicer.add_contours(gm, levels=[.51], colors=["r"])
    if not wm_filename is None:
        _slicer.add_contours(wm_filename, levels=[.51], colors=["g"])
    if not csf_filename is None:
        _slicer.add_contours(csf_filename, levels=[.51], colors=['b'])

    # misc
    _slicer.title(title, size=12, color='w', alpha=0)

    if report_path not in [False, None]:
        fig = plt.gcf()
        svg_plot = _plot_to_svg(fig)
    else:
        svg_plot = None

    if not output_filename is None:
        plt.savefig(output_filename, bbox_inches='tight', dpi=200,
                    facecolor="k", edgecolor="k")
        if close:
            plt.close()
            
    return svg_plot
