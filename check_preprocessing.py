"""
This script checks that all the dataset is OK, using basic statistics
1. write an image of the coefficient of variation for each voxel
   in each session and subject
2. plot the time course of cross-voxel median coefficient of variation
   in each subject
3. plot the time courses of motion parameters

XXX TODO: plot CV maps (code disabled below in plot_cv_tc)
XXX TODO: use other statistics for QA (besides CV)
"""
import os
import glob
import numpy as np
from scipy import stats
import pylab as pl

import nibabel as ni
from nipy.labs import compute_mask_files
from nipy.labs import viz
from joblib import Memory as CheckPreprocMemory

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 6}

pl.rc('font', **font)

EPS = np.finfo(float).eps


def plot_spm_motion_parameters(parameter_file, subject_id=None, format="png",
                               title=None):
    """ Plot motion parameters obtained with SPM software

    Parameters
    ----------
    parameter_file: string,
                    path of file containing the motion parameters
    """
    # load parameters
    motion = np.loadtxt(parameter_file)
    motion[:, 3:] *= (180. / np.pi)

    # do plotting
    pl.figure()
    pl.plot(motion)
    if not title is None:
        pl.title(title)
    elif not subject_id is None:
        pl.title("subject: %s" % subject_id)
    pl.xlabel('time(scans)')
    pl.legend(('Ty', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'), prop={'size': 6})
    pl.ylabel('Estimated motion (mm/degrees)',
              fontsize=10)

    # dump image unto disk
    img_filename = "%s.%s" % (parameter_file, format)
    pl.savefig(img_filename, bbox_inches="tight", dpi=200)

    return img_filename


def check_mask(data):
    """
    Create the data mask and check that the volume is reasonable

    Parameters
    ----------
    data: string: path of some input data

    returns
    -------
    mask_array: array of shape ni.load(data).get_shape(),
                the binary mask

    """
    mask_array = compute_mask_files(epi_data[0])
    affine = ni.load(epi_data[0]).get_affine()
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


def my_plot_cv_tc(epi_data, subject_id, mask_array=None):
    nim = ni.load(epi_data)
    affine = nim.get_affine()
    assert len(nim.shape) == 4

    # get the data
    data = nim.get_data()
    thr = stats.scoreatpercentile(data.ravel(), 7)
    data[data < thr] = thr

    # compute cv
    cv = compute_cv(data, mask_array)

    pl.plot(cv, label=subject_id)
    pl.legend()
    pl.xlabel('time(scans)')
    pl.ylabel('Median coefficient of variation')
    pl.axis('tight')

    output_filename = epi_data.sub(".nii", "").sub(".gz", "") + "_cv.png"
    pl.savefig(output_filename)

    return output_filename


def plot_cv_tc(epi_data, session_ids, subject_id, output_dir, do_plot=True,
               write_image=True, mask=True, bg_image=False,
               cv_plot_outfiles=None, cv_tc_plot_outfile=None, plot_diff=False,
               title=None):
    """
    Compute coefficient of variation of the data and plot it

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
    assert len(epi_data) == len(session_ids)
    if not cv_plot_outfiles:
        assert len(cv_plot_outfiles) == len(epi_data)

    cv_tc_ = []
    if isinstance(mask, basestring):
        mask_array = ni.load(mask).get_data() > 0
    elif mask == True:
        mask_array = compute_mask_files(epi_data[0])
    else:
        mask_array = None
    count = 0
    for (session_id, fmri_file) in zip(session_ids, epi_data):
        nim = ni.load(fmri_file)
        affine = nim.get_affine()
        if len(nim.shape) == 4:
            # get the data
            data = nim.get_data()
        else:
            raise RuntimeError, "expecting 4D image, got %iD" % len(nim.shape)

        # compute the CV for the session
        cache_dir = os.path.join(output_dir, "CV")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        mem = CheckPreprocMemory(cachedir=cache_dir, verbose=5)
        cv = mem.cache(compute_cv)(data, mask_array)

        if write_image:
            # write an image
            data_dir = os.path.dirname(fmri_file)
            ni.save(ni.Nifti1Image(cv, affine),
                 os.path.join(data_dir, 'cv_%s.nii' % session_id))
            if bg_image == False:
                viz.plot_map(cv,
                             affine,
                             threshold=.01,
                             cmap=pl.cm.spectral,
                             black_bg=True,
                             title="subject: %s, session: %s" %\
                                 (subject_id, session_id))
            elif isinstance(bg_image, basestring):
                anat, anat_affine = (
                    ni.load(bg_image).get_data(),
                    ni.load(bg_image).get_affine())
            else:
                anat, anat_affine = data.mean(-1), affine
                viz.plot_map(cv, affine, threshold=.01,
                             cmap=pl.cm.spectral,
                             anat=anat, anat_affine=anat_affine)

            if not cv_plot_outfiles is None:
                pl.savefig(cv_plot_outfiles[count])
                count += 1

        # compute the time course of cv
        cv_tc_sess = np.median(
            np.sqrt((data[mask_array > 0].T /
                     data[mask_array > 0].mean(-1) - 1) ** 2), 1)

        cv_tc_.append(cv_tc_sess)
    cv_tc = np.concatenate(cv_tc_)

    if do_plot:
        # plot the time course of cv for different subjects
        stuff = [cv_tc]
        legends = ['Median Coefficient of Variation']
        if plot_diff:
            diff_cv_tc = np.hstack(([0], np.diff(cv_tc)))
            stuff.append(diff_cv_tc)
            legends.append('Differential Coefficent of Variation')
        legends = tuple(legends)
        pl.plot(np.vstack(stuff).T)
        pl.legend(legends)

        pl.xlabel('time(scans)')
        pl.ylabel('Median Coefficient of Variation')
        pl.axis('tight')

        if title:
            pl.title(title)

        if not cv_tc_plot_outfile is None:
            pl.savefig(cv_tc_plot_outfile)

    return cv_tc


def plot_registration(reference, coregistered,
                        title="untitled coregistration!",
                        cut_coords=None,
                        output_filename=None):
    """
    QA for coregistration: plots a coregistered source as bg/contrast
    for the reference image. This way, see the similarity between the
    two images.

    """
    # set cut_coords
    if cut_coords is None:
        cut_coords = (-2, -28, 17)  # XXX FIXME: determine this!

    # plot the coregistered image
    coregistered_img = ni.load(coregistered)
    coregistered_data = coregistered_img.get_data()
    coregistered_affine = coregistered_img.get_affine()
    slicer = viz.plot_anat(anat=coregistered_data,
                           anat_affine=coregistered_affine,
                           black_bg=True,
                           cmap=pl.cm.spectral,
                           cut_coords=cut_coords,
                           )

    # overlap the reference image
    reference_img = ni.load(reference)
    reference_data = reference_img.get_data()
    reference_affine = reference_img.get_affine()
    slicer.edge_map(reference_data, reference_affine)

    # misc
    slicer.title(title, size=12, color='w',
                 alpha=0)

    if not output_filename is None:
        pl.savefig(output_filename, dpi=200, bbox_inches='tight',
                     facecolor="k",
                     edgecolor="k")


def plot_segmentation(img_filename, gm_filename, wm_filename, csf_filename,
                      output_filename=None, cut_coords=None,
                      slicer='ortho',
                      title='GM + WM + CSF segmentation'):
    """
    Plot a contour mapping of the GM, WM, and CSF of a subject's anatomical.

    """
    if cut_coords is None:
        cut_coords = (-2, -28, 17)

    if slicer in ['x', 'y', 'z']:
        cut_coords = (cut_coords['xyz'.index(slicer)],)

    # plot img
    img = ni.load(img_filename)
    anat = img.get_data()
    anat_affine = img.get_affine()
    _slicer = viz.plot_anat(anat, anat_affine, cut_coords=cut_coords,
                            slicer=slicer,
                            black_bg=True, cmap=pl.cm.spectral)

    # draw a GM contour map
    gm = ni.load(gm_filename)
    gm_template = gm.get_data()
    gm_affine = gm.get_affine()
    _slicer.contour_map(gm_template, gm_affine, levels=[.51], colors=["r"])

    # draw a WM contour map
    wm = ni.load(wm_filename)
    wm_template = wm.get_data()
    wm_affine = wm.get_affine()
    _slicer.contour_map(wm_template, wm_affine, levels=[.51], colors=["g"])

    # draw a CSF contour map
    csf = ni.load(csf_filename)
    csf_template = csf.get_data()
    csf_affine = csf.get_affine()
    _slicer.contour_map(csf_template, csf_affine, levels=[.51], colors=['b'])

    # misc
    _slicer.title(title, size=10, color='w',
                 alpha=0)
    # pl.legend(("WM", "CSF", "GM"))

    if not output_filename is None:
        pl.savefig(output_filename, bbox_inches='tight', dpi=200,
                   facecolor="k",
                   edgecolor="k")


# Demo
if __name__ == '__main__':
    data_dir = '/tmp/va100099/fmri/'
    session_ids = ['000007', '000009']
    subject_id = 'va100099'
    spm_motion_parameters = glob.glob(os.path.join(data_dir, 'rp*.txt'))
    epi_data = glob.glob(os.path.join(data_dir, 'wr*.nii'))
    epi_data.sort()

    plot_spm_motion_parameters(spm_motion_parameters)
    check_mask(epi_data[0])
    plot_cv_tc(epi_data, session_ids, subject_id)
