"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

import os
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.purepython_preproc_utils import do_subject_preproc
from pypreprocess._spike.pipeline_comparisons import execute_spm_auditory_glm

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
subject_data = do_subject_preproc(subject_data,
                                  do_stc=True,
                                  fwhm=[8] * 3
                                  )

# run glm on data
execute_spm_auditory_glm(subject_data)
