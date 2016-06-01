"""
Automatic configuration of MATLAB/SPM back-end
"""
# author: Elvis DOHMATOB, Yanick SCHWARZ

import os
import re
import glob
import distutils
import logging

from nipype.interfaces import matlab
from nipype.interfaces import spm
import nipype


LOG_FILE = os.path.abspath('./configure_spm.log')

_logger = logging.getLogger(__name__)
_logger.propagate = False
_logger.setLevel(logging.DEBUG)
console_logger = logging.StreamHandler()
console_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
console_logger.setFormatter(formatter)
_logger.addHandler(console_logger)

try:
    with open(LOG_FILE, 'w') as log_f:
        pass
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)

except IOEroor as e:
    _logger.warn('could not open log file for writing: {}'.format(LOG_FILE))

# default paths
DEFAULT_SPM_DIRS = [os.path.join(os.environ.get('HOME', '/'), 'opt/spm8/'),
                    '/i2bm/local/spm8',
                    os.path.join(os.environ.get('HOME', '/'),
                    "spm8-standalone/spm8_mcr")]

DEFAULT_MATLAB_EXECS = ["/neurospin/local/bin/matlab"]

DEFAULT_SPM_MCRS = [
    os.path.join(os.environ.get('HOME', '/'), 'opt/spm8/spm8.sh'),
    "/i2bm/local/bin/spm8",
    "/storage/workspace/usr/local/spm8",
    os.path.join(os.environ.get('HOME', '/'), "spm8")]

def _check_nipype_version():
    if distutils.version.LooseVersion(
            nipype.__version__).version < [0, 9]:
        return False
    return True

def _find_or_warn(loc, check, msg=None, warner=_logger.warn):
    if loc is None:
        return False
    if check(loc):
        return True
    if msg is not None:
        warner(msg)
    return False

def _find_dep_loc(specified_loc=None, exported_name=None, default_locs=None,
                  check=os.path.exists, msg_prefix=''):

    if _find_or_warn(specified_loc, check, ''.join(
            [msg_prefix, ' specified location: "{}"'.format(specified_loc)])):
        return specified_loc

    exported_loc = os.environ.get(exported_name)
    if _find_or_warn(exported_loc, check, ''.join(
            [msg_prefix, ' exported location: "{}"'.format(exported_loc)])):
        return exported_loc

    for def_loc in default_locs:
        if _find_or_warn(def_loc, check, ''.join(
                [msg_prefix, ' default location: "{}"'.format(def_loc)]),
                         warner=_logger.debug):
            return def_loc

    return None

def _is_executable(file_name):
    return os.access(file_name, os.X_OK)

def _find_spm_mcr(spm_mcr):
    spm_mcr = _find_dep_loc(
        spm_mcr, 'SPM_MCR', DEFAULT_SPM_MCRS,
        check=_is_executable,
        msg_prefix='SPM MCR is not executable or could not be found in')
    if spm_mcr is not None:
        if _check_nipype_version():
            _logger.info('setting SPM MCR backend: '
                         'set matlab executable path to "{}" '
                         'and "use_mcr" to True'.format(spm_mcr))
            spm.SPMCommand.set_mlab_paths(
                matlab_cmd='{} run script'.format(spm_mcr),
                use_mcr=True)
            return spm_mcr
        _logger.warn(
            'nipype version {} too old.'
            ' No support for precompiled SPM'.format(nipype.__version__))
        return None

def _find_matlab_exec_and_spm_dir(spm_dir, matlab_exec):
    matlab_exec = _find_dep_loc(
        matlab_exec,
        'MATLAB_EXEC', DEFAULT_MATLAB_EXECS,
        check=_is_executable,
        msg_prefix='matlab executable file could not be found in')
    if matlab_exec is not None:
        spm_dir = _find_dep_loc(
            spm_dir, 'SPM_DIR', DEFAULT_SPM_DIRS,
            check=os.path.isdir,
            msg_prefix='SPM directory could not be found in')
        if spm_dir is not None:
            _logger.info('setting default matlab command to be "{}"'.format(
                matlab_exec))
            matlab.MatlabCommand.set_default_matlab_cmd(matlab_exec)
            _logger.info('setting matlab default paths to "{}"'.format(spm_dir))
            matlab.MatlabCommand.set_default_paths(spm_dir)
            return spm_dir
    return None

def _configure_spm(spm_dir=None, matlab_exec=None, spm_mcr=None):
    """
    -look for an SPM MCR precompiled binary.
        if found, return its parent directory.
    -if not found, look for a Matlab executable and an SPM directory.
        if found, return the SPM directory
    if neither is found, return None.

    when looking for a directory or a file,
    we examine the corresponding passed argument first,
    then the corresponding environment variable,
    then in some default locations.
    """
    spm_mcr = _find_spm_mcr(spm_mcr)
    if spm_mcr is not None:
        spm_dir =  os.path.abspath(os.path.dirname(spm_mcr))
        return spm_dir

    spm_dir = _find_matlab_exec_and_spm_dir(spm_dir, matlab_exec)
    if spm_dir is not None:
        return spm_dir

    _logger.error('could not find SPM_MCR precompiled executable'
                  ' nor a pair (Matlab executable and SPM directory)')
    return

def _get_version_spm(spm_dir):
    """ Return current SPM version
    """
    _, spm_version = os.path.split(spm_dir)
    if not spm_version:
        _loggers.warn('Could not figure out SPM version!'
                     ' (spm dir: "{}")'.format(spm_dir))
        return
    return spm_version
