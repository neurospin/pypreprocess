"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: single_subject_pipeline.py demo

"""

import os
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.purepython_preproc_utils import do_subject_preproc
import nibabel

# fetch data
sd = fetch_spm_auditory_data(os.path.join(os.path.abspath('.'),
                                          "spm_auditory"))

# preproc data
for flag in xrange(3):
    for pre_concat in [True, False]:
        # pack data into dict, the format understood by the pipeleine
        subject_data = {
            'n_sessions': 1,  # number of sessions
            'func': [sd.func],  # functional (BOLD) images 1 item/session
            'anat': sd.anat,  # anatomical (structural) image
            'subject_id': 'sub001',
            'output_dir': os.path.abspath('spm_auditory_preproc')
            }
        if pre_concat:
            subject_data['func'] = [nibabel.concat_images(sess_func,
                                                       check_affines=False)
                                 for sess_func in subject_data['func']]
            do_subject_preproc(subject_data,
                       stc=True,
                       fwhm=[8] * 3,
                       write_output_images=flag
                       )
