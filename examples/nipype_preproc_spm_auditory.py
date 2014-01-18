"""
:Module: nipype_preproc_spm_auditory
Synopsis: Minimal script for preprocessing single-subject data
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

# standard imports
import sys
import os

# import API for preprocessing business
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc

# input data-grabber for SPM Auditory (single-subject) data
from pypreprocess.datasets import fetch_spm_auditory_data

# file containing configuration for preprocessing the data
this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
jobfile = os.path.join(this_dir, "spm_auditory_preproc.ini")

# set dataset dir
if len(sys.argv) > 1:
    dataset_dir = sys.argv[1]
else:
    dataset_dir = os.path.join(this_dir, "spm_auditory")


# fetch spm auditory data
fetch_spm_auditory_data(dataset_dir)

# preprocess the data
results = do_subjects_preproc(jobfile, dataset_dir=dataset_dir)
assert len(results) == 1
