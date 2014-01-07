"""
:Module: nipype_preproc_nyu_rest
Synopsis: Minimal script for preprocessing single-subject data
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

# standard imports
import sys
import os

# import API for preprocessing business
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc

# input data-grabber for SPM Auditory (single-subject) data
from pypreprocess.datasets import fetch_nyu_rest

# file containing configuration for preprocessing the data
this_dir = os.path.dirname(sys.argv[0])
jobfile = os.path.join(os.path.dirname(sys.argv[0]),
                       "nyu_rest_preproc.conf")

# set dataset dir
if len(sys.argv) > 1:
    dataset_dir = sys.argv[1]
else:
    dataset_dir = os.path.join(this_dir, "nyu_rest")


# # fetch spm auditory data
# fetch_nyu_rest(data_dir=dataset_dir)

# preprocess the data
results = do_subjects_preproc(jobfile, dataset_dir=dataset_dir)
assert len(results) == 1
