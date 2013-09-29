"""
:Synopsis: Example usage of purepython_preroc_pipeline module, step-by-step
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.io_utils import load_4D_img, load_vol
from pypreprocess.slice_timing import fMRISTC
from pypreprocess.realign import MRIMotionCorrection
from pypreprocess.coreg import Coregister
from joblib import Memory
import os

# create cache
mem = Memory('/tmp/stepwise_cache', verbose=100)


def _cached(f):
    """
    Sandbox for executing cached function calls.

    """

    return mem.cache(f)

# fetch input data
sd = fetch_spm_auditory_data(os.path.join(
        os.environ['HOME'],
        "/home/elvis/CODE/datasets/spm_auditory"))
n_sessions = 1  # this dataset has 1 session (i.e 1 fMRI acquisiton or run)

# ouput dict
output = {'func': [sd.func],  # one fMRI 4D datum per session
          'anat': sd.anat, 'n_sessions': n_sessions}

# load the data
output['func'] = [_cached(load_4D_img)(sess_func)
                  for sess_func in output['func']]
output['anat'] = _cached(load_vol)(output['anat'])

##################################
# Slice-Timing Correction (STC)
##################################
for sess_func, sess_id in zip(output['func'], xrange(n_sessions)):
    # session fit
    fmristc = _cached(fMRISTC().fit)(raw_data=sess_func)

    # session transform
    output['func'][sess_id] = _cached(fmristc.transform)(sess_func)

###########################
# Motion Correction (MC)
###########################
mrimc = _cached(MRIMotionCorrection(n_sessions=n_sessions, quality=1.).fit)(
    output['func'])
mc_output = _cached(mrimc.transform)(reslice=True)

output["func"] = mc_output['realigned_images']
output["realignment_parameters"] = mc_output['realignment_parameters']

###################
# Coregistration
###################
coreg = _cached(Coregister().fit)(output["anat"], output['func'][0])
output['func'] = [_cached(coreg.transform)(sess_func)
                   for sess_func in output['func']]

print output
