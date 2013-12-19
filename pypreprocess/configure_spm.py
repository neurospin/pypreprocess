import os
import glob
import nipype.interfaces.matlab as matlab
from nipype.interfaces import spm

# configure MATLAB exec
MATLAB_EXEC = "/neurospin/local/matlab/bin/matlab"
if not os.path.exists(MATLAB_EXEC):
    m_choices = glob.glob("/neurospin/local/matlab/R*/bin/matlab")
    if m_choices:
        MATLAB_EXEC = m_choices[0]
if 'MATLAB_EXEC' in os.environ:
    MATLAB_EXEC = os.environ['MATLAB_EXEC']
assert os.path.exists(MATLAB_EXEC), \
    "nipype_preproc_smp_utils: MATLAB_EXEC: %s, \
doesn't exist; you need to export MATLAB_EXEC" % MATLAB_EXEC
matlab.MatlabCommand.set_default_matlab_cmd(MATLAB_EXEC)

# configure matlab SPM
SPM_DIR = '/i2bm/local/spm8'
if 'SPM_DIR' in os.environ:
    SPM_DIR = os.environ['SPM_DIR']
assert os.path.exists(SPM_DIR), \
    "nipype_preproc_smp_utils: SPM_DIR: %s,\
 doesn't exist; you need to export SPM_DIR" % SPM_DIR
matlab.MatlabCommand.set_default_paths(SPM_DIR)
