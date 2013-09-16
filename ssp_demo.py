"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

# import goodies
from external.nilearn.datasets import (
    fetch_spm_auditory_data,
    fetch_spm_multimodal_fmri_data
    )
from spike.single_subject_pipeline import do_subject_preproc
from pipeline_comparisons import (execute_spm_auditory_glm,
                                  execute_spm_multimodal_fmri_glm
                                  )

# # ##############################################
# # # SPM Multimodal fMRI (faces vs. scrambled)
# # ##############################################

# # fetch data
# sd = fetch_spm_multimodal_fmri_data(
#     "/home/elvis/CODE/datasets/spm_multimodal_fmri")

# # pack data into dict, the format understood by the pipeleine
# subject_data = {
#     'n_sessions': 2,  # number of sessions
#     'func': [sd.func1, sd.func2],  # functional (BOLD) images 1 item/session
#     'anat': sd.anat,  # anatomical (structural) image
#     'subject_id': 'sub001',
#     'output_dir': 'spm_multimodal_fmri_preproc',
#     'trials_ses1': sd.trials_ses1,
#     'trials_ses2': sd.trials_ses2
#     }

# # run preproc pipeline
# preproc_output  = do_subject_preproc(
#     subject_data,
#     slice_order='descending',
#     # fwhm=[10, 10, 10],
#     write_preproc_output_images=True,
#     concat=True
#     )

# # run GLM on preprocesses data
# execute_spm_multimodal_fmri_glm(preproc_output)

################################
# SPM single-subject Auditory
################################

# fetch data
sd = fetch_spm_auditory_data("/home/elvis/CODE/datasets/spm_auditory")

# pack data into dict, the format understood by the pipeleine
subject_data = {
    'n_sessions': 1,  # number of sessions
    'func': [sd.func],  # functional (BOLD) images 1 item/session
    'anat': sd.anat,  # anatomical (structural) image
    'subject_id': 'sub001',
    'output_dir': 'spm_auditory_preproc'
    }

# run preproc pipeline
preproc_output  = do_subject_preproc(
    subject_data,
    fwhm=[10, 10, 10],
    write_preproc_output_images=True,
    concat=True
    )

# run GLM on preprocesses data
execute_spm_auditory_glm(preproc_output)
