"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

from pypreprocess.datasets import fetch_spm_auditory
from pypreprocess.purepython_preproc_utils import do_subject_preproc

# fetch data
sd = fetch_spm_auditory()
sd.output_dir = "/tmp/sub001"
sd.func = [sd.func]

# preproc data
do_subject_preproc(sd.__dict__, concat=False, coregister=True, stc=True,
                   tsdiffana=True, realign=True, report=True)
