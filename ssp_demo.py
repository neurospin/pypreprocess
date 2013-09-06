"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

# import goodies
from external.nilearn.datasets import fetch_spm_auditory_data
import spike.single_subject_pipeline as ssp

# fetch data
sd = fetch_spm_auditory_data("/home/elvis/CODE/datasets/spm_auditory")

# pack data into dict, the format understood by the pipeleine
subject_data = {'n_sessions': 1,  # number of sessions
                'func': [sd.func],  # functional (BOLD) images 1 item/session
                'anat': sd.anat,  # anatomical (structural) image
                'subject_id': 'sub001',
                'output_dir': 'spm_auditory_preproc'
                }

# run preproc pipeline
preproc_output  = ssp.do_subject_preproc(subject_data,
                                         write_preproc_output_images=True
                                         )

# run GLM on preprocesses data
ssp.execute_spm_auditory_glm(preproc_output)
