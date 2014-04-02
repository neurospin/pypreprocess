"""
:Module: configure_spm
:Synopsis: Configures MATLAB/SPM back-end
:Author: DOHMATOB Elvis Dopgima

"""

import os
import glob
import distutils


import nipype.interfaces.matlab as matlab
from nipype.interfaces import spm
import nipype

from io_utils import _expand_path

DEFAULT_SPM_DIR = '/i2bm/local/spm8'
DEFAULT_MATLAB_EXEC = "/neurospin/local/matlab/bin/matlab"


def configure_spm(matlab_exec=None, spm_dir=None):
    origin_spm_dir = spm_dir

    # configure SPM
    if 'SPM_DIR' in os.environ:
        spm_dir = os.environ['SPM_DIR']

    if not spm_dir is None:
        spm_dir = _expand_path(spm_dir)

    if spm_dir is None or not os.path.isdir(spm_dir):
        if spm_dir:
            print "Path %s doesn't exist" % spm_dir
        print "spm_dir defaulting to %s" % DEFAULT_SPM_DIR
        spm_dir = DEFAULT_SPM_DIR

    assert os.path.exists(spm_dir), (
        "Can't find SPM path '%s'! You should export SPM_DIR=/path/to/"
        "your/SPM/root/dir" % spm_dir)

    if (distutils.version.LooseVersion(nipype.__version__).version
                >= [0, 9] and
            os.path.exists('/i2bm/local/bin/spm8')
            and origin_spm_dir is None):
        if 'SPM8_MCR' in os.environ:
            matlab_cmd = '%s run script' % os.environ['SPM8_MCR']
        else:
            matlab_cmd = '/i2bm/local/bin/spm8 run script'
        spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
    else:
        matlab.MatlabCommand.set_default_paths(spm_dir)

    # configure MATLAB
    if 'MATLAB_EXEC' in os.environ:
        matlab_exec = os.environ['MATLAB_EXEC']

    if not matlab_exec is None:
        matlab_exec = _expand_path(matlab_exec)

    if matlab_exec is None or not os.path.exists(matlab_exec):
        if matlab_exec:
            print "Path %s doesn't exist" % matlab_exec
        matlab_exec = DEFAULT_MATLAB_EXEC

        if not os.path.exists(matlab_exec):
            m_choices = glob.glob("/neurospin/local/matlab/R*/bin/matlab")
            if m_choices:
                matlab_exec = m_choices[0]

    assert os.path.exists(matlab_exec), (
        "Can't find matlab path: '%s' ! You should export MATLAB_EXEC="
        "/path/to/your/matlab/exec, doesn't exist; you need to export "
        "matlab_exec" % matlab_exec)

    matlab.MatlabCommand.set_default_matlab_cmd(matlab_exec)

    return spm_dir, matlab_exec
