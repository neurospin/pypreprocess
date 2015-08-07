# -*- coding: utf-8 -*-
"""
Demo script for anatomical MRI segmentation with SPM.

It demos segmentations on SPM single-subject auditory.

@author: Mehdi RAHIM
"""

import os
from pypreprocess.subject_data import SubjectData
from pypreprocess.datasets import fetch_spm_auditory
from pypreprocess.nipype_preproc_spm_utils import _do_subject_segment
from pypreprocess.reporting.check_preprocessing import plot_segmentation


OUTPUT_DIR = 'segmentation_demos_output'
def _spm_auditory_subject_data():
    """ Fetching auditory example into SubjectData Structure
    """
    subject_data = fetch_spm_auditory()
    subject_data['func'] = None
    subject_data.output_dir = os.path.join(OUTPUT_DIR)
    return SubjectData(**subject_data)


# Fetch and generate the subject_data structure
subject_data = _spm_auditory_subject_data()

# Segment the GM, WM and the CSF
subject_data = _do_subject_segment(subject_data, caching=True, report=False,
                                   hardlink_output=False)

# Overlay the tissues with T1 acquisition
plot_segmentation(img=subject_data['anat'],
                  gm_filename=subject_data['gm'],
                  wm_filename=subject_data['wm'],
                  csf_filename=subject_data['csf'])
