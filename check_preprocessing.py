"""
:Module: check_preprocessing
:Synopsis: module for generating post-preproc plots (registration,
segmentation, etc.) using the viz module from nipy.labs.
:Author: bertrand thirion, dohmatob elvis dopgima

"""

import os
import numpy as np
import pylab as pl

import nibabel

from nipy.labs import compute_mask_files
from nipy.labs import viz
import joblib
from io_utils import do_3Dto4D_merge, compute_mean_3D_image

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

    """

    # load parameters
    motion = np.loadtxt(parameter_file)
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


def check_mask(epi_data):
    """
    Create the data mask and check that the volume is reasonable

    Parameters
    ----------
    data: string: path of some input data

    returns
    -------
    mask_array: array of shape nibabel.load(data).get_shape(),
                the binary mask

    """
    mask_array = compute_mask_files(epi_data[0])
    affine = nibabel.load(epi_data[0]).get_affine()
    vol = np.abs(np.linalg.det(affine)) * mask_array.sum() / 1000
    print 'The estimated brain volume is: %f cm^3, should be 1000< <2000' % vol
    return mask_array


def compute_cv(data, mask_array=None):
    if mask_array is not None:
        cv = .0 * mask_array
        cv[mask_array > 0] = data[mask_array > 0].std(-1) /\
            (data[mask_array > 0].mean(-1) + EPS)
    else:
        cv = data.std(-1) / (data.mean(-1) + EPS)

    return cv


def plot_cv_tc(epi_data, session_ids, subject_id,
               do_plot=True,
               write_image=True, mask=True, bg_image=False,
               plot_diff=True,
               cv_tc_plot_outfile=None):
    """ Compute coefficient of variation of the data and plot it

    Parameters
    ----------
    epi_data: list of strings, input fMRI 4D images
    session_ids: list of strings of the same length as epi_data,
                 session indexes (for figures)
    subject_id: string, id of the subject (for figures)
    do_plot: bool, optional,
             should we plot the resulting time course
    write_image: bool, optional,
                 should we write the cv image
    mask: bool or string, optional,
          (string) path of a mask or (bool)  should we mask the data
    bg_image: bool or string, optional,
              (string) pasth of a background image for display or (bool)
              should we compute such an image as the mean across inputs.
              if no, an MNI template is used (works for normalized data)
    """

    cv_tc_ = []
    if isinstance(mask, basestring):
        mask_array = nibabel.load(mask).get_data() > 0
    elif mask == True:
        mask_array = compute_mask_files(epi_data[0])
    else:
        mask_array = None
    for (session_id, fmri_file) in zip(session_ids, epi_data):
        if type(fmri_file) is str:
            data_dir = os.path.dirname(fmri_file)
        else:
            data_dir = os.path.dirname(fmri_file[0])

        nim = nibabel.load(do_3Dto4D_merge(fmri_file))
        affine = nim.get_affine()
        if len(nim.shape) == 4:
            # get the data
            data = nim.get_data()
        else:
            raise TypeError("Expecting 4D image!")
            pass

        # compute the CV for the session
        cache_dir = os.path.join(data_dir, "CV")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        mem = joblib.Memory(cachedir=cache_dir, verbose=5)
        cv = mem.cache(compute_cv)(data, mask_array=mask_array)

        if write_image:
            # write an image
            nibabel.save(nibabel.Nifti1Image(cv, affine),
                         os.path.join(data_dir, 'cv_%s.nii' % session_id))
            if bg_image == False:
                viz.plot_map(cv, affine, threshold=.01, cmap=viz.cm.cold_hot)
            else:
                if isinstance(bg_image, basestring):
                    _tmp = nibabel.load(bg_image)
                    anat, anat_affine = (
                        _tmp.get_data(),
                        _tmp.get_affine())
                else:
                    anat, anat_affine = data.mean(-1), affine
                viz.plot_map(cv, affine, threshold=.01, cmap=viz.cm.cold_hot,
                             anat=anat, anat_affine=anat_affine)
        # compute the time course of cv
        cv_tc_sess = np.median(
            np.sqrt((data[mask_array > 0].T /
                     data[mask_array > 0].mean(-1) - 1) ** 2), 1)

        cv_tc_.append(cv_tc_sess)
    cv_tc = np.concatenate(cv_tc_)

    if do_plot:
        # plot the time course of cv for different subjects
        pl.figure()
        pl.plot(cv_tc, label=subject_id)
        pl.legend()
        pl.xlabel('time(scans)')
        pl.ylabel('Median coefficient of variation')
        pl.axis('tight')

        if not cv_tc_plot_outfile is None:
            pl.savefig(cv_tc_plot_outfile,
                       bbox_inches="tight", dpi=200)

    return cv_tc


# def plot_cv_tc(epi_data, subject_id, session_id="UNKNOWN_SESSION",
#                output_dir=None, threshold=0.01,
#                write_image=True, mask=True, bg_image=False,
#                cv_plot_outfile=None, cv_tc_plot_outfile=None, do_plot=True,
#                plot_diff=True):

#     """
#     Compute coefficient of variation of the data and plot it

#     Parameters
#     ----------
#     epi_data: epi data path(s) (string or list of strings)
#     subject_id : string
#                  id of subject under inspection
#     session_id:  string
#                  session id
#     write_image: bool, optional,
#                  should we write the cv image
#     mask: bool or string, optional,
#           (string) path of a mask or (bool)  should we mask the data
#     bg_image: bool or string, optional,
#               (string) pasth of a background image for display or (bool)
#               should we compute such an image as the mean across inputs.
#               if no, an MNI template is used (works for normalized data)
#     """

#     # sanity
#     if type(epi_data) is str:
#         epi_data = [epi_data]

#     if output_dir is None:
#         _tmp = epi_data
#         if not type(epi_data) is str:
#             _tmp = epi_data[0]
#         output_dir = os.path.dirname(_tmp)

#     if isinstance(mask, basestring):
#         mask_array = nibabel.load(mask).get_data() > 0
#     elif mask == True:
#         mask_array = compute_mask_files(epi_data[0])
#     else:
#         mask_array = None

#     nim = nibabel.load(do_3Dto4D_merge(epi_data))
#     affine = nim.get_affine()
#     if len(nim.shape) == 4:
#         # get the data
#         data = nim.get_data()
#     else:
#         raise RuntimeError("expecting 4D image, got %iD" % len(nim.shape))

#     # compute the CV for the session
#     cache_dir = os.path.join(output_dir, "CV")
#     if not os.path.exists(cache_dir):
#         os.makedirs(cache_dir)
#     mem = joblib.Memory(cachedir=cache_dir, verbose=5)
#     cv = mem.cache(compute_cv)(data, mask_array)

#     # write an image
#     if write_image:
#         nibabel.save(nibabel.Nifti1Image(cv, affine),
#                      os.path.join(output_dir, 'cv_%s.nii' % session_id))

#     # plot cv proper
#     if bg_image == False:
#         title = "CV map thresholded at %s" % (100 * threshold) + "%"
#         viz.plot_map(cv,
#                      affine,
#                      threshold=threshold,
#                      # cut_coords=(-2, -28, 17),
#                      cmap=pl.cm.spectral,
#                      title=title,
#                      )
#     else:
#         if isinstance(bg_image, basestring):
#             anat, anat_affine = (
#                 nibabel.load(bg_image).get_data(),
#                 nibabel.load(bg_image).get_affine())
#         else:
#             anat, anat_affine = (data.mean(-1), affine)
#         viz.plot_map(
#             cv, affine, threshold=threshold,
#             cut_coords=(-10, -28, 17),
#             cmap=pl.cm.spectral,
#             anat=anat, anat_affine=anat_affine,
#             title=title
#             )

#     if not cv_plot_outfile is None:
#         pl.savefig(cv_plot_outfile, dpi=200, bbox_inches='tight',
#                    facecolor="k",
#                    edgecolor="k")

#     # compute the time course of the cv
#     cv_tc = np.median(
#         np.sqrt((data[mask_array > 0].T /
#                  data[mask_array > 0].mean(-1) - 1) ** 2), 1)

#     # plot the time course of the cv
#     if do_plot:
#         pl.figure()
#         stuff = [cv_tc]
#         legends = ['Median Coefficient of Variation']
#         if plot_diff:
#             diff_cv_tc = np.hstack(([0], np.diff(cv_tc)))
#             stuff.append(diff_cv_tc)
#             legends.append('Differential Median Coefficent of Variation')
#         legends = tuple(legends)
#         pl.plot(np.vstack(stuff).T)
#         pl.legend(legends, loc="center", prop={'size': 12})

#         pl.xlabel('time(scans)')
#         pl.ylabel('Median Coefficient of Variation')
#         pl.axis('tight')

#         if not cv_tc_plot_outfile is None:
#             pl.savefig(cv_tc_plot_outfile, bbox_inches="tight", dpi=200)

#     return cv_tc


def plot_registration(reference_img, coregistered_img,
                      title="untitled coregistration!",
                      cut_coords=None,
                      slicer='ortho',
                      cmap=None,
                      output_filename=None):
    """Plots a coregistered source as bg/contrast for the reference image

    Parameters
    ----------
    reference_img: string
        path to reference (background) image

    coregistered_img: string
        path to other image (to be compared with reference)

    slicer: string (optional, defaults to 'ortho')
        slicer param to pass to the nipy.labs.viz.plot_??? APIs

    cmap: matplotlib colormap object (optional, defaults to spectral)
        colormap to user for plots

    output_filename: string (optional)
        path where plot will be stored

    """

    if cmap is None:
        cmap = pl.cm.spectral

    if cut_coords is None:
        cut_coords = (-10, -28, 17)

    if slicer in ['x', 'y', 'z']:
        cut_coords = (cut_coords['xyz'.index(slicer)],)

    # plot the coregistered image
    if hasattr(coregistered_img, '__len__'):
        coregistered_img = compute_mean_3D_image(coregistered_img)
    # XXX else i'm assuming a nifi object ;)
    coregistered_data = coregistered_img.get_data()
    coregistered_affine = coregistered_img.get_affine()
    _slicer = viz.plot_anat(
        anat=coregistered_data,
        anat_affine=coregistered_affine,
        cmap=cmap,
        cut_coords=cut_coords,
        slicer=slicer,
        black_bg=True,
        )

    # overlap the reference image
    if hasattr(reference_img, '__len__'):
        reference_img = compute_mean_3D_image(reference_img)
    # XXX else i'm assuming a nifi object ;)
    reference_data = reference_img.get_data()
    reference_affine = reference_img.get_affine()
    _slicer.edge_map(reference_data, reference_affine)

    # misc
    _slicer.title(title, size=12, color='w',
                  alpha=0)

    if not output_filename is None:
        try:
            pl.savefig(output_filename, dpi=200, bbox_inches='tight',
                       facecolor="k",
                       edgecolor="k")
        except AttributeError:
            # XXX TODO: handy this case!!
            pass


def plot_segmentation(img, gm_filename, wm_filename=None,
                      csf_filename=None,
                      output_filename=None, cut_coords=None,
                      slicer='ortho',
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
    if cmap is None:
        cmap = pl.cm.spectral

    if cut_coords is None:
        cut_coords = (-10, -28, 17)

    if slicer in ['x', 'y', 'z']:
        cut_coords = (cut_coords['xyz'.index(slicer)],)

    # plot img
    if hasattr(img, '__len__'):
        img = compute_mean_3D_image(img)
    # XXX else i'm assuming a nifi object ;)
    anat = img.get_data()
    anat_affine = img.get_affine()
    _slicer = viz.plot_anat(
        anat, anat_affine, cut_coords=cut_coords,
        slicer=slicer,
        cmap=cmap,
        black_bg=True,
        )

    # draw a GM contour map
    gm = nibabel.load(gm_filename)
    gm_template = gm.get_data()
    gm_affine = gm.get_affine()
    _slicer.contour_map(gm_template, gm_affine, levels=[.51], colors=["r"])

    # draw a WM contour map
    if not wm_filename is None:
        wm = nibabel.load(wm_filename)
        wm_template = wm.get_data()
        wm_affine = wm.get_affine()
        _slicer.contour_map(wm_template, wm_affine, levels=[.51], colors=["g"])

    # draw a CSF contour map
    if not csf_filename is None:
        csf = nibabel.load(csf_filename)
        csf_template = csf.get_data()
        csf_affine = csf.get_affine()
        _slicer.contour_map(
            csf_template, csf_affine, levels=[.51], colors=['b'])

    # misc
    _slicer.title(title, size=12, color='w',
                 alpha=0)
    # pl.legend(("WM", "CSF", "GM"), loc="lower left", ncol=len(cut_coords))

    if not output_filename is None:
        pl.savefig(output_filename, bbox_inches='tight', dpi=200,
                   facecolor="k",
                   edgecolor="k")


# Demo
if __name__ == '__main__':
    pass  # XXX placeholder for demo code
