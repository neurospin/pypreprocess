"""
Automatic configuration of MATLAB/SPM back-end
"""
# author: Elvis DOHMATOB, Yanick SCHWARZ

import os
import re
import glob
import distutils
import logging
import itertools

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

def _find_or_warn(loc, check, msg=None, warner=_logger.warn, recursive=False):
    if loc is None:
        return None
    loc = os.path.abspath(os.path.expanduser(loc))
    if check(loc):
        return loc
    if(recursive and os.path.isdir(loc)):
        for (dirpath, dirnames, filenames) in os.walk(loc):
            for name in itertools.chain(dirnames, filenames):
                found_path = os.path.join(dirpath, name)
                if check(found_path):
                    return found_path
    if msg is not None:
        warner(msg)
    return None

def _find_dep_loc(specified_loc=None, exported_name=None, default_locs=None,
                  check=os.path.exists, msg_prefix='', recursive=False):

    message = '{} specified location: "{}"'.format(msg_prefix, specified_loc)
    found = _find_or_warn(specified_loc, check, msg=message,
                          recursive=recursive)
    if found is not None:
        return found

    exported_loc = os.environ.get(exported_name)
    message = '{} exported location: "{}"'.format(msg_prefix, exported_loc)
    found = _find_or_warn(exported_loc, check, msg=message, recursive=recursive)
    if found is not None:
        return found

    for def_loc in default_locs:
        message = '{} default location: "{}"'.format(msg_prefix, def_loc)
        found = _find_or_warn(def_loc, check, msg=message,
                              recursive=recursive, warner=_logger.debug)
        if found is not None:
            return found

    return None

def _is_executable(file_name):
    return os.access(file_name, os.X_OK)

def _find_spm_mcr(spm_mcr):
    spm_mcr = _find_dep_loc(
        spm_mcr, 'SPM_MCR', DEFAULT_SPM_MCRS,
        check=_is_executable,
        msg_prefix='SPM MCR is not executable or could not be found in')
    if spm_mcr is None:
        return None

    if _check_nipype_version():
        _logger.info('setting SPM MCR backend: '
                     'set matlab executable path to "{}" '
                     'and "use_mcr" to True'.format(spm_mcr))
        spm.SPMCommand.set_mlab_paths(
            matlab_cmd='{} run script'.format(spm_mcr),
            use_mcr=True)
        DEFAULT_SPM_DIRS.insert(0, os.path.dirname(spm_mcr))
        return spm_mcr
    _logger.warn(
        'nipype version {} too old.'
        ' No support for precompiled SPM'.format(nipype.__version__))
    return None

def _is_spm_dir(dir_path):
    return(os.path.isdir(dir_path) and 'tpm' in os.listdir(dir_path))

def _find_spm_dir(spm_dir):
    return _find_dep_loc(
        spm_dir, 'SPM_DIR', DEFAULT_SPM_DIRS,
        check=_is_spm_dir,
        msg_prefix='SPM directory could not be found in',
        recursive=True)

def _find_spm_mcr_and_spm_dir(spm_mcr, spm_dir):
    spm_mcr = _find_spm_mcr(spm_mcr)
    if spm_mcr is None:
        return None

    spm_dir = _find_spm_dir(spm_dir)
    if spm_dir is not None:
        _logger.info('found SPM directory to be "{}"'.format(spm_dir))
        return spm_dir
    return None

def _find_matlab_exec_and_spm_dir(spm_dir, matlab_exec):
    matlab_exec = _find_dep_loc(
        matlab_exec,
        'MATLAB_EXEC', DEFAULT_MATLAB_EXECS,
        check=_is_executable,
        msg_prefix='matlab executable file could not be found in')
    if matlab_exec is None:
        return None

    spm_dir = _find_spm_dir(spm_dir)
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
    found_spm_dir = _find_spm_mcr_and_spm_dir(spm_mcr, spm_dir)
    if found_spm_dir is not None:
        return found_spm_dir

    found_spm_dir = _find_matlab_exec_and_spm_dir(spm_dir, matlab_exec)
    if found_spm_dir is not None:
        return found_spm_dir

    _logger.error('could not find SPM_MCR precompiled executable'
                  ' nor a pair (Matlab executable and SPM directory)')
    return

def _get_version_spm(spm_dir):
    """ Return current SPM version
    """
    _, spm_version = os.path.split(spm_dir)
    if not spm_version:
        _logger.warn('Could not figure out SPM version!'
                     ' (spm dir: "{}")'.format(spm_dir))
        return
    return spm_version
