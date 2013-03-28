"""
:Module: slice_timing
:Author: dohmatob elvis dopgima
:Synopsis: module for slice timing business

"""

import os
import nipy
import nibabel as ni
from nipy.algorithms.registration.groupwise_registration import FmriRealign4d
import numpy as np
import joblib
from nipy.io.nifti_ref import nifti2nipy
from io_utils import is_3D


def do_slicetiming_and_motion_correction(func,
                                         output_dir=None,
                                         **fmrirealign4d_kwargs):
    """
    Function to do slice-timing and motion correction together.

    Parameters
    ----------
    func: string or list of strings
       filename or list of filenames for images to be realigned

    output_dir: string (optional, default None)
        output directory to which all output files will be written,
        if write_output option is set

    **fmrirealign4D_kwargs:
        options to be passed to
        ``nipy.algorithms.registration.groupwise_registration.FmriRealign4D``
        constructor

        tr: float (TR for images)
        slice_order ['ascending' | 'descending' | etc.]
        time_interp: [True | False]


    """

    # sanity
    single = False
    if isinstance(func, basestring):
        single = True
        func = [func]

    if not output_dir is None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # instantiate FmriRealign4d object
    fourD_input_fmri_files = func

    # load images
    runs = []
    for x in fourD_input_fmri_files:
        if not isinstance(x, basestring):
            assert is_3D(x[0]), "Expecting list of 3D images, x!"
            raise IOError("Slice-timing not yet implemented for 3D images!")
        else:
            img = nipy.load_image(x)
        runs.append(img)

    R = FmriRealign4d(runs,
                      **fmrirealign4d_kwargs)

    # estimate transforms
    R.estimate()

    # apply transforms
    realigned_runs = R.resample()

    # sanity checks
    assert len(runs) == len(realigned_runs)

    # collect output stuff
    rp_files = []
    fourD_output_fmri_files = []
    for j in xrange(len(realigned_runs)):
        realignment_params = np.array(
            [np.hstack((R._transforms[j][i].translation,
                        R._transforms[j][i].rotation))
                    for i in xrange(len(R._transforms[j]))])

        if not output_dir is None:
            # save realigned image
            input_dtype = ni.load(fourD_input_fmri_files[j]).get_data_dtype()

            input_file_basename = os.path.basename(
                fourD_input_fmri_files[j]).split(".")
            output_file_basename = input_file_basename[
                0] + "_nipy_realigned" + "." + input_file_basename[1]
            output_img_path = os.path.join(output_dir,
                                           output_file_basename)

            if not os.path.exists(output_img_path):
                nipy.save_image(
                    realigned_runs[j],
                    output_img_path,
                    dtype_from=input_dtype)

            fourD_output_fmri_files.append(output_img_path)

            # save motion params
            rp_file = os.path.join(output_dir,
                                   "rp_run_%i.txt" % j)

            np.savetxt(rp_file, realignment_params)

            rp_files.append(rp_file)
        else:
            fourD_output_fmri_files.append(realigned_runs[j])
            rp_files.append(realignment_params)

    # return outputs
    if single:
        fourD_output_fmri_files = fourD_output_fmri_files[0]

    return fourD_output_fmri_files, rp_files
