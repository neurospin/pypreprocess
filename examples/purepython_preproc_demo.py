"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

import os
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.purepython_preproc_utils import do_subject_preproc
from pypreprocess.subject_data import SubjectData
import nibabel

# fetch data
sd = fetch_spm_auditory_data(os.path.join(os.path.abspath('.'),
                                          "spm_auditory"))
sd.output_dir = "/tmp/sub001"
sd.func = [sd.func]

# preproc data
do_subject_preproc(sd.__dict__, concat=False, coregister=True,
                   stc=True, cv_tc=True, realign=True,
                   report=True)
