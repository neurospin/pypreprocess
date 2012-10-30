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

from nibabel import load, Nifti1Image, save
from nipy.labs import compute_mask_files
from nipy.labs import viz

EPS = np.finfo(float).eps

def subplot_spm_motion_parameters(parameter_file, subject_id=None, subplot=None):
    """ Plot motion parameters obtained with SPM software

    Parameters
    ----------
    Parameter_file: string,
                    path of file containing the motion parameters
    subject_id: string, optional,
                subject id
    subplot: optional
           where to attach the image
    """
    if subplot is None:
        subplot = pl.figure().add_subplot(1, 1, 1)

    motion = np.loadtxt(parameter_file)
    motion[:, 3:] *= (180. / np.pi)

    subplot.plot(motion)
    # if subject_id is not None:
    #     subplot.set_title("subject: %s" % subject_id)
    subplot.set_xlabel('time(scans)')
    subplot.legend(('Ty', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz'))

    subplot.set_ylabel('Estimated motion (mm/degrees)')


def check_mask(data):
    """
    Create the data mask and check that the volume is reasonable

    Parameters
    ----------
    data: string: path of some input data

    returns
    -------
    mask_array: array of shape load(data).get_shape(),
                the binary mask

    """
    mask_array = compute_mask_files(epi_data[0])
    affine = load(epi_data[0]).get_affine()
    vol = np.abs(np.linalg.det(affine)) * mask_array.sum() / 1000
    print 'The estimated brain volume is: %f cm^3, should be 1000< <2000' % vol
    return mask_array


def subplot_cv_tc(epi_data, session_ids, subject_id, do_plot=True,
               write_image=True, mask=True, bg_image=False, subplot=None):
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

    subplot: optional
           where to attach the image
    """
    cv_tc_ = []
    if isinstance(mask, basestring):
        mask_array = load(mask).get_data() > 0
    elif mask == True:
        mask_array = compute_mask_files(epi_data[0])
    else:
        mask_array = None
    for (session_id, fmri_file) in zip(session_ids, epi_data):
        nim = load(fmri_file)
        affine = nim.get_affine()
        if len(nim.shape) == 4:
            # get the data
            data = nim.get_data()
            thr = stats.scoreatpercentile(data.ravel(), 7)
            data[data < thr] = thr

        else:
            # fixme: todo
            pass

        # compute the CV for the session
        if mask_array is not None:
            cv = .0 * mask_array
            cv[mask_array > 0] = data[mask_array > 0].std(-1) /\
                (data[mask_array > 0].mean(-1) + EPS)
        else:
            cv = data.std(-1) / (data.mean(-1) + EPS)

        if write_image:
            # write an image
            data_dir = os.path.dirname(fmri_file)
            save(Nifti1Image(cv, affine),
                 os.path.join(data_dir, 'cv_%s.nii' % session_id))
            if bg_image == False:
                # viz.plot_map(cv,
                #              affine,
                #              threshold=.01, # XXX why this threshold ?
                #              cmap=pl.cm.spectral,
                #              black_bg=True,
                #              title="subject: %s, session: %s" %\
                #                  (subject_id, session_id))
                pass
            elif isinstance(bg_image, basestring):
                anat, anat_affine = (
                    load(bg_image).get_data(),
                    load(bg_image).get_affine())
            else:
                anat, anat_affine = data.mean(-1), affine
                # slicer = viz.plot_map(cv, affine, threshold=.01, cmap=pl.cm.spectral,
                #                       anat=anat, anat_affine=anat_affine)
                pass

        # compute the time course of cv
        cv_tc_sess = np.median(
            np.sqrt((data[mask_array > 0].T /
                     data[mask_array > 0].mean(-1) - 1) ** 2), 1)

        cv_tc_.append(cv_tc_sess)
    cv_tc = np.concatenate(cv_tc_)

    if do_plot:
        # plot the time course of cv for different subjects
        if subplot is None:
            subplot = pl.figure().add_subplot(1, 1, 1)
        subplot.plot(cv_tc, label=subject_id)
        subplot.legend()
        subplot.set_xlabel('time(scans)')
        subplot.set_ylabel('Median coefficient of variation')
        subplot.axis('tight')

    return cv_tc

# Demo
if __name__ == '__main__':
    data_dir = '/tmp/va100099/fmri/'
    session_ids = ['000007', '000009']
    subject_id = 'va100099'
    spm_motion_parameters = glob.glob(os.path.join(data_dir, 'rp*.txt'))
    epi_data = glob.glob(os.path.join(data_dir, 'wr*.nii'))
    epi_data.sort()

    subplot_spm_motion_parameters(spm_motion_parameters)
    check_mask(epi_data[0])
    subplot_cv_tc(epi_data, session_ids, subject_id)
    pl.show()
