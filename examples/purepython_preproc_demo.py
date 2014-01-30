"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

import os
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.purepython_preproc_utils import do_subject_preproc

# fetch data
sd = fetch_spm_auditory_data(os.path.join(os.path.abspath('.'),
                                          "spm_auditory"))

# pack data into dict, the format understood by the pipeleine
subject_data = {'n_sessions': 1,  # number of sessions
                'func': [sd.func],  # functional (BOLD) images 1 item/session
                'anat': sd.anat,  # anatomical (structural) image
                'subject_id': 'sub001',
                'output_dir': os.path.abspath('spm_auditory_preproc')
                }

# preproc data
do_subject_preproc(subject_data,
                   stc=True,
                   fwhm=[8] * 3,
                   write_output_images=0
                   )
