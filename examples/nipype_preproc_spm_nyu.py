"""
Preprocessing of NYU rest data.
"""

# standard imports
import sys
import os

# import API for preprocessing business
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc

# input data-grabber for SPM Auditory (single-subject) data
from pypreprocess.datasets import fetch_nyu_rest

# file containing configuration for preprocessing the data
jobfile = os.path.join(os.path.dirname(sys.argv[0]),
                       "nyu_rest_preproc.ini")


# fetch spm auditory data
sd = fetch_nyu_rest()

# preprocess the data
dataset_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        sd.anat_skull[0]))))
results = do_subjects_preproc(jobfile, dataset_dir=dataset_dir)
