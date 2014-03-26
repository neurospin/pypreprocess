"""
:Synopsis:  Step-by-step example usage of purepython_preroc_pipeline module
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.slice_timing import fMRISTC
from pypreprocess.realign import MRIMotionCorrection
from pypreprocess.coreg import Coregister
from pypreprocess.external.joblib import Memory
import os

# create cache
mem = Memory('/tmp/stepwise_cache', verbose=100)

# fetch input data
sd = fetch_spm_auditory_data(os.path.join(
        os.environ['HOME'],
        "CODE/datasets/spm_auditory"))
n_sessions = 1  # this dataset has 1 session (i.e 1 fMRI acquisiton or run)

do_subject_preproc(sd.__dict__(), concat=False, coregister=True,
                   stc=True, cv_tc=True, realign=True,
                   report=True)
