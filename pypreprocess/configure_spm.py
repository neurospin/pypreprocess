"""
Automatic configuration of MATLAB/SPM back-end
"""
# author: Elvis DOHMATOB, Yanick SCHWARZ

import os
import re
import glob
import warnings
import distutils
import nipype.interfaces.matlab as matlab
from nipype.interfaces import spm
import nipype

# default paths
DEFAULT_SPM_DIRS = ['/i2bm/local/spm8'][:0]
DEFAULT_MATLAB_EXECS = ["/neurospin/local/bin/matlab"]
DEFAULT_SPM_MCRS = ["/i2bm/local/bin/spm8"]


def _configure_spm(spm_dir=None, matlab_exec=None, spm_mcr=None):
    """Configure SPM backend."""
    # sanitize input spm_dir
    if not spm_dir is None:
        if spm_dir is None or not os.path.isdir(spm_dir):
            warnings.warn("Specified spm_dir '%s' is not a directory!" % (
                    spm_dir))
            spm_dir = None

    # try using default SPM directories
    if spm_dir is None:
        for spm_dir in DEFAULT_SPM_DIRS:
            if os.path.isdir(spm_dir):
                break
        else:
            spm_dir = None

    if spm_dir is None:
        # set spm_dir to SPM_DIR exported variable
        if "SPM_DIR" in os.environ:
            if not os.path.isdir(os.environ["SPM_DIR"]):
                warnings.warn("Exported SPM_DIR '%s' is not a directory!" % (
                        os.environ["SPM_DIR"]))
            else:
                spm_dir = os.environ["SPM_DIR"]
                # sanitize matlab_exec input
                if not matlab_exec is None:
                    if matlab_exec is None or not os.path.isdir(matlab_exec):
                        warnings.warn(
                            "Specified matlab_exec '%s' is not a file!" % (
                                matlab_exec))
                        matlab_exec = None

    # sanitize input matlab_exec
    if not matlab_exec is None:
        if matlab_exec is None or not os.path.isfile(matlab_exec):
            warnings.warn("Specified matlab_exec '%s' is not a file!" % (
                    matlab_exec))
            matlab_exec = None

    # try using default MATLAB paths
    if matlab_exec is None:
        for matlab_exec in DEFAULT_MATLAB_EXECS:
            if os.path.isfile(matlab_exec):
                break
        else:
            matlab_exec = None

    if matlab_exec is None:
        # set matlab_exec to MATLAB_EXEC exported variable
        if "MATLAB_EXEC" in os.environ:
            if not os.path.isdir(os.environ["MATLAB_EXEC"]):
                warnings.warn(
                    "Exported MATLAB_EXEC'%s' is not a file!" % (
                        os.environ["MATLAB_EXEC"]))
            else:
                matlab_exec = os.environ["MATLAB_EXEC"]

    # configure SPM and MATLAB backend proper
    if not None in [spm_dir, matlab_exec]:
        cmd = "matlab.MatlabCommand.set_default_matlab_cmd('%s')" % matlab_exec
        warnings.warn("Setting matlab backend with cmd: %s" % cmd)
        eval(cmd)
        cmd = "matlab.MatlabCommand.set_default_paths('%s')" % spm_dir
        warnings.warn("Setting SPM backend with cmd: %s" % cmd)
        eval(cmd)
    else:
        # there is a problem with SPM / MATLAB; try to use precompiled SPM
        # instead
        if distutils.version.LooseVersion(
            nipype.__version__).version >= [0, 9]:
            if not spm_mcr is None:
                if spm_mcr is None or not os.path.exists(spm_mcr):
                    warnings.warn(
                        "Specified spm_mcr '%s' doesn't exist!" % (
                            spm_mcr))
                    spm_mcr = None

        # try using default MATLAB paths
        if spm_mcr is None:
            for spm_mcr in DEFAULT_SPM_MCRS:
                if os.path.isfile(spm_mcr):
                    break
            else:
                spm_mcr = None

            if spm_mcr is None:
                # set spm_mcr to SPM_MCR exported variable
                if "SPM_MCR" in os.environ:
                    if not os.path.isdir(os.environ["SPM_MCR"]):
                        warnings.warn(
                            "Exported SPM_MCR'%s' is not a file!" % (
                                os.environ["SPM_MCR"]))
                    else:
                        spm_mcr = os.environ["SPM_MCR"]

            # configure SPM MCR backend proper
            if not spm_mcr is None:
                cmd = ("spm.SPMCommand.set_mlab_paths("
                       "matlab_cmd='%s run script', use_mcr=True)" % (
                        spm_mcr))
                warnings.warn("Setting SPM MCR backend with cmd: %s" % cmd)
                eval(cmd)

                # infer directory containing SPM templates, tpms, etc.
                fd = open(spm_mcr, 'r')
                code = fd.read()
                fd.close()
                m = re.search("SPM.+?_STANDALONE_HOME=(.+)", code)
                if m:
                    spm_dir = m.group(1)
                else:
                    spm_dir = os.path.dirname(spm_mcr)
                tpm_path = glob.glob(os.path.join(spm_dir,
                                                  "*_mcr/spm*/tpm"))[0]
                spm_dir = os.path.dirname(tpm_path)
            else:
                warnings.warn('Failed to configure SPM!')
                return
        else:
            warnings.warn(
                "Nipype version %s too old. No support for precompiled SPM!")

    return spm_dir
