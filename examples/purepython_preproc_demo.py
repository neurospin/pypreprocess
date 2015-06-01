"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

from pypreprocess.datasets import fetch_spm_multimodal_fmri
from pypreprocess.purepython_preproc_utils import do_subject_preproc

# fetch data
sd = fetch_spm_multimodal_fmri()
sd.output_dir = "/tmp/sub001"
sd.func = [sd.func1, sd.func2]

# preproc data
do_subject_preproc(sd.__dict__, concat=False, coregister=True, stc=True,
                   tsdiffana=True, realign=True, report=True, reslice=True)
