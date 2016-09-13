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
import matplotlib.pyplot as plt


OUTPUT_DIR = 'segmentation_demos_output'
def _spm_auditory_subject_data():
    """ Fetching auditory example into SubjectData Structure
    """
    subject_data = fetch_spm_auditory()
    subject_data['func'] = None
    base_dir = os.path.dirname(subject_data['anat'])
    subject_data.output_dir = os.path.join(base_dir, OUTPUT_DIR)
    return SubjectData(**subject_data)


# Fetch and generate the subject_data structure
print('Fetching Auditory Dataset')
subject_data = _spm_auditory_subject_data()

# Segment the GM, WM and the CSF
print('Segmentation with SPM')
subject_data = _do_subject_segment(subject_data, caching=True, report=False,
                                   hardlink_output=False)
print('Segmentation saved in : %s' % subject_data.output_dir)

# Overlay the tissues with T1 acquisition
plot_segmentation(img=subject_data['anat'],
                  gm_filename=subject_data['gm'],
                  wm_filename=subject_data['wm'],
                  csf_filename=subject_data['csf'])
plt.show()
