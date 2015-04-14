"""
:Module: check_preprocessing
:Synopsis: module for generating post-preproc plots (registration,
segmentation, etc.) using the viz module from nipy.labs.
:Author: bertrand thirion, dohmatob elvis dopgima

"""

import os
import tempfile
import numpy as np
import pylab as pl
import nibabel
from nilearn.plotting import plot_img, plot_stat_map
from nilearn.image import reorder_img, mean_img
from ..external import joblib
from ..io_utils import load_4D_img, load_specific_vol

EPS = np.finfo(float).eps


def plot_spm_motion_parameters(parameter_file, title=None,
                               output_filename=None):
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
        parameter_file, basestring) else parameter_file[..., :6]

    motion[:, 3:] *= (180. / np.pi)

    # do plotting
    pl.figure()
    pl.plot(motion)
    if not title is None:
        pl.title(title, fontsize=10)
    pl.xlabel('time(scans)', fontsize=10)
    pl.legend(('Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'), prop={"size": 12},
              loc="upper left", ncol=2)
    pl.ylabel('Estimated motion (mm/degrees)', fontsize=10)

    # dump image unto disk
    if not output_filename is None:
        pl.savefig(output_filename, bbox_inches="tight", dpi=200)
        pl.close()


def compute_cv(data, mask_array=None):
    if mask_array is not None:
        cv = .0 * mask_array
        cv[mask_array > 0] = data[mask_array > 0].std(-1) /\
            (data[mask_array > 0].mean(-1) + EPS)
    else:
        cv = data.std(-1) / (data.mean(-1) + EPS)

    return cv


def plot_cv_tc(epi_imgs, session_ids, subject_id,
               do_plot=True, write_image=True, _output_dir=None,
               cv_tc_plot_outfile=None, **kwargs):
    """ Compute coefficient of variation of the data and plot it

    Parameters
    ----------
    epi_imgs: list of strings, input fMRI 4D images
    session_ids: list of strings of the same length as epi_imgs,
                 session indexes (for figures)
    subject_id: string, id of the subject (for figures)
    do_plot: bool, optional,
             should we plot the resulting time course
    write_image: bool, optional,
                 should we write the cv image
    mask: bool or string, optional,
          (string) path of a mask or (bool)  should we mask the data
    **kwargs:
        kwargs for plot_stat_map API
    """
    if _output_dir is None:
        if not cv_tc_plot_outfile is None:
            _output_dir = os.path.dirname(cv_tc_plot_outfile)
        else:
            _output_dir = tempfile.mkdtemp()

    lengths = []
    cv_tc = []
    for session_id, fmri_file in zip(session_ids, epi_imgs):
        nim = load_4D_img(fmri_file)
        affine = nim.get_affine()
        if len(nim.shape) == 4:
            data = nim.get_data()
        else:
            raise TypeError("Expecting 4D image!")

        # compute the CV for the session
        cache_dir = os.path.join(_output_dir, "CV")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        mem = joblib.Memory(cachedir=cache_dir, verbose=5)
        cv = nibabel.Nifti1Image(mem.cache(compute_cv)(data), affine)

        # XXX nilearn complains about rotations in affine, etc.
        cv = reorder_img(cv, resample="continuous")

        if write_image:
            # write an image
            cv.to_filename(os.path.join(_output_dir, 'cv_%s.nii' % session_id))
        plot_stat_map(cv, threshold=.01, **kwargs)

        # compute the time course of cv
        data = data.reshape((-1, data.shape[-1]))
        lengths.append(data.shape[-1])
        cv_tc_sess = np.median(np.sqrt((data.T / data.mean(axis=-1) - 1) ** 2),
                               axis=-1)
        cv_tc.append(cv_tc_sess)
    cv_tc = np.concatenate(cv_tc)

    # plot CV time-course
    if do_plot:
        pl.figure()
        pl.plot(cv_tc, label=subject_id)
        pl.legend()
        pl.xlabel('time(scans)')
        pl.ylabel('Median coefficient of variation')
        aux = 0.
        for l in lengths[:-1]:
            pl.axvline(aux + l, linestyle="--", c="k")
            aux += l
        pl.axis('tight')
        if not cv_tc_plot_outfile is None:
            pl.savefig(cv_tc_plot_outfile,
                       bbox_inches="tight", dpi=200)
            pl.close()

    return cv_tc


def plot_registration(reference_img, coregistered_img,
                      title="untitled coregistration!",
                      cut_coords=None,
                      display_mode='ortho',
                      cmap=None,
                      output_filename=None):
    """Plots a coregistered source as bg/contrast for the reference image

    Parameters
    ----------
    reference_img: string
        path to reference (background) image

    coregistered_img: string
        path to other image (to be compared with reference)

    display_mode: string (optional, defaults to 'ortho')
        display_mode param to pass to the nipy.labs.viz.plot_??? APIs

    cmap: matplotlib colormap object (optional, defaults to spectral)
        colormap to user for plots

    output_filename: string (optional)
        path where plot will be stored

    """

    # sanity
    if cmap is None:
        cmap = pl.cm.gray  # registration QA always gray cmap!

    if cut_coords is None:
        cut_coords = (-10, -28, 17)

    if display_mode in ['x', 'y', 'z']:
        cut_coords = (cut_coords['xyz'.index(display_mode)],)

    # plot the coregistered image
    if hasattr(coregistered_img, '__len__'):
        coregistered_img = load_specific_vol(coregistered_img, 0)[0]

    # XXX nilearn complains about rotations in affine, etc.
    coregistered_img = reorder_img(coregistered_img, resample="continuous")

    _slicer = plot_img(coregistered_img, cmap=cmap, cut_coords=cut_coords,
              display_mode=display_mode, black_bg=True)

    # overlap the reference image
    if hasattr(reference_img, '__len__'):
        reference_img = load_specific_vol(reference_img, 0)[0]

    # XXX nilearn complains about rotations in affine, etc.
    reference_img = reorder_img(reference_img, resample="continuous")

    _slicer.add_edges(reference_img)

    # misc
    _slicer.title(title, size=12, color='w', alpha=0)

    if not output_filename is None:
        try:
            pl.savefig(output_filename, dpi=200, bbox_inches='tight',
                       facecolor="k",
                       edgecolor="k"
                       )
            pl.close()
        except AttributeError:
            # XXX TODO: handle this case!!
            pass


def plot_segmentation(img, gm_filename, wm_filename=None,
                      csf_filename=None,
                      output_filename=None, cut_coords=None,
                      display_mode='ortho',
                      cmap=None,
                      title='GM + WM + CSF segmentation'):
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
        cmap = pl.cm.gray
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
    if not output_filename is None:
        pl.savefig(output_filename, bbox_inches='tight', dpi=200,
                   facecolor="k",
                   edgecolor="k")
        pl.close()
