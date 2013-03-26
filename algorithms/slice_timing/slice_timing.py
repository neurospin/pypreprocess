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


def do_slicetiming_and_motion_correction(output_dir,
                                         func,
                                         write_output=False,
                                         return_realignment_params=False,
                                         **fmrirealign4d_kwargs):
    """
    Function to do slice-timing and motion correction together.

    XXX Undocumented API!!!

    """

    fourD_input_fmri_files = func
    runs = [nipy.load_image(x) for x in fourD_input_fmri_files]
    R = FmriRealign4d(runs,
                      **fmrirealign4d_kwargs)

    # estimate transformations
    R.estimate()

    # apply transformations
    realigned_runs = R.resample()

    # sanity checks
    assert len(runs) == len(realigned_runs)

    if return_realignment_params:
        rp_files = []
        for j in xrange(len(realigned_runs)):
            rp_file = os.path.join(output_dir,
                                   "rp_run_%i.txt" % j)

            realignment_params = np.array(
                [np.hstack((R._transforms[j][i].translation,
                            R._transforms[j][i].rotation))
                        for i in xrange(len(R._transforms[j]))])

            np.savetxt(rp_file, realignment_params)

            rp_files.append(rp_file)

    if not write_output:
        fourD_output_fmri_files = realigned_runs
    else:
        # save realigned images to disk
        fourD_output_fmri_files = []
        for j in xrange(len(fourD_input_fmri_files)):
            output_img_path = os.path.join(output_dir,
                                           "func_run_%i.nii" % j)

            nipy.save_image(
                realigned_runs[j],
                output_img_path,
                dtype_from=ni.load(
                    fourD_input_fmri_files[j]).get_data_dtype())

            fourD_output_fmri_files.append(output_img_path)

        if return_realignment_params:
            return fourD_output_fmri_files, rp_files
        else:
            return fourD_output_fmri_files
