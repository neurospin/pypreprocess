"""
:Synopsis:  Step-by-step example usage of purepython_preroc_pipeline module
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.slice_timing import fMRISTC
from pypreprocess.realign import MRIMotionCorrection
from pypreprocess.coreg import Coregister
from joblib import Memory
import os

# create cache
mem = Memory('/tmp/stepwise_cache', verbose=100)

# fetch input data
sd = fetch_spm_auditory_data(os.path.join(
        os.environ['HOME'],
        "CODE/datasets/spm_auditory"))
n_sessions = 1  # this dataset has 1 session (i.e 1 fMRI acquisiton or run)

# ouput dict
output = {'func': [sd.func],  # one fMRI 4D datum per session
          'anat': sd.anat, 'n_sessions': n_sessions}

##################################
# Slice-Timing Correction (STC)
##################################
for sess_func, sess_id in zip(output['func'], xrange(n_sessions)):
    # session fit
    fmristc = mem.cache(fMRISTC().fit)(raw_data=sess_func)

    # session transform
    output['func'][sess_id] = mem.cache(fmristc.transform)(sess_func)

###########################
# Motion Correction (MC)
###########################
mrimc = mem.cache(MRIMotionCorrection(n_sessions=n_sessions, quality=1.).fit)(
    output['func'])
mc_output = mem.cache(mrimc.transform)(reslice=True)

output["func"] = mc_output['realigned_images']
output["realignment_parameters"] = mc_output['realignment_parameters']

###################
# Coregistration
###################
coreg = mem.cache(Coregister().fit)(output["anat"], output['func'][0])
output['func'] = [mem.cache(coreg.transform)(sess_func)
                  for sess_func in output['func']]

print output
