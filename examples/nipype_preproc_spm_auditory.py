"""
:Module: nipype_preproc_spm_auditory
Synopsis: Minimal script for preprocessing single-subject data
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

# standard imports
import sys
import os

# import API for preprocessing business
from pypreprocess.nipype_preproc_spm_utils_bis import (do_subjects_preproc,
                                                       SubjectData
                                                       )

# input data-grabber for SPM Auditory (single-subject) data
from pypreprocess.datasets import fetch_spm_auditory_data

# sanitize command-line
if len(sys.argv)  < 3:
    print ("\r\nUsage: python %s <spm_auditory_MoAEpilot_dir>"
           " <output_dir>\r\n") % sys.argv[0]
    sys.exit(1)

# set i/o directories
DATA_DIR = os.path.abspath(sys.argv[1])
OUTPUT_DIR = os.path.abspath(sys.argv[2])

# fetch spm auditory data
sd = fetch_spm_auditory_data(DATA_DIR)

# make subject data
subject_data = SubjectData()
subject_data.func = sd.func
subject_data.anat = sd.anat
subject_data.subject_id = "sub001"
subject_data.output_dir = os.path.join(OUTPUT_DIR, subject_data.subject_id)

# preprocess the data
results = do_subjects_preproc(
    [subject_data],
    output_dir=OUTPUT_DIR,
    fwhm=0.,  # 8mm isotropic Gaussian kernel
    dataset_id="SPM single-subject auditory",
    dataset_description=('<a href="http://www.fil.ion.ucl.ac.uk/spm/data'
                         '/auditory/">SPM auditory dataset</a>.</p>'),
    # do_normalize=False,
    last_stage=True
    )
