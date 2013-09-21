"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

import os
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.purepython_preproc_utils import do_subject_preproc

# fetch data
sd = fetch_spm_auditory_data(os.path.join(os.environ['HOME'],
                                          "CODE/datasets/spm_auditory"))

# pack data into dict, the format understood by the pipeleine
subject_data = {'n_sessions': 1,  # number of sessions
                'func': [sd.func],  # functional (BOLD) images 1 item/session
                'anat': sd.anat,  # anatomical (structural) image
                'subject_id': 'sub001',
                'output_dir': 'spm_auditory_preproc'
                }

# run preproc pipeline
preproc_output  = do_subject_preproc(subject_data,
                                     concat=True
                                     )
