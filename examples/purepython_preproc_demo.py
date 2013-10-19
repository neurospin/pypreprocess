"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

import os
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.purepython_preproc_utils import do_subject_preproc
from pypreprocess._spike.pipeline_comparisons import execute_spm_auditory_glm
from pypreprocess.nipype_preproc_spm_utils import (
    do_subject_preproc as dsp,
    SubjectData
    )

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
                                     do_stc=True,
                                     # fwhm=[8] * 3
                                     )

sd = SubjectData()
sd.subject_id = preproc_output['subject_id']
sd.func = preproc_output['func']
sd.output_dir = os.path.join(preproc_output['output_dir'], sd.subject_id)
preproc_output['func'] = dsp(
    preproc_output,
    do_realign=False,
    do_coreg=False,
    do_segment=False,
    do_normalize=False,
    do_report=False,
    fwhm=[8] * 3).func

# run glm
execute_spm_auditory_glm(preproc_output)
