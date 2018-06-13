"""
Automatic configuration of MATLAB/SPM back-end
"""
# author: Elvis DOHMATOB, Yanick SCHWARZ, Jerome DOCKES

import os
import re
import glob
import distutils
import logging
import itertools
from functools import partial

from nipype.interfaces import matlab
from nipype.interfaces import spm
import nipype


def prepare_logging(log_stream=True, log_file="./configure_spm.log"):
    """ Define the logger handlers.

    Parameters
    ----------
    log_stream: bool, default True
        if True create a stream INFO handler.
    log_file: str, default './configure_spm.log'
        if specified create a file DEBUG handler, if None don't create such
        handler.
    """
    _logger = logging.getLogger("pypreprocess")
    if(_logger.handlers):
        return _logger
    _logger.propagate = False
    _logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    if log_stream:
        console_logger = logging.StreamHandler()
        console_logger.setLevel(logging.INFO)
        console_logger.setFormatter(formatter)
        _logger.addHandler(console_logger)
    if log_file is not None:
        try:
            with open(log_file, 'a') as log_f:
                pass
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)

        except IOEroor as e:
            _logger.warning(
                'could not open log file for writing: {}'.format(log_file))

    return _logger


_logger = logging.getLogger("pypreprocess")

# TODO: read this from config?
_ACCEPT_SPM_MCR_WITH_UNKNOWN_VERSION = False

# TODO: read this from config?
_ACCEPT_SPM_MCR_WITH_AMBIGUOUS_VERSION = True

_SPM_DEFAULTS = {}

_SPM_DEFAULTS['version_numbers'] = [12, 8]

_SPM_DEFAULTS['spm_mcr_template'] = [
    os.path.join(os.environ.get('HOME'),
                 'opt/spm{VERSION_NB}/spm{VERSION_NB}.sh'),
    "/opt/spm{VERSION_NB}/spm{VERSION_NB}.sh",
    "/i2bm/local/bin/spm{VERSION_NB}",
    "/storage/workspace/usr/local/spm{VERSION_NB}",
    os.path.join(os.environ.get('HOME'), "spm{VERSION_NB}")]

_SPM_DEFAULTS['spm_dir_template'] = [
    os.path.join(os.environ.get('HOME'), 'opt/spm{VERSION_NB}/'),
    "/opt/spm{VERSION_NB}/",
    '/i2bm/local/spm{VERSION_NB}',
    os.path.join(os.environ.get('HOME'),
                 "spm{VERSION_NB}-standalone/spm{VERSION_NB}_mcr")]

_SPM_DEFAULTS['matlab_exec'] = ["/neurospin/local/bin/matlab"]

# give priority to SPM_MCR env variable before specific versions like SPM8_MCR.
_SPM_DEFAULTS['spm_mcr_env_template'] = ['SPM_MCR', 'SPM{VERSION_NB}_MCR']

# here we already know the version of the mcr, so prefer the specific dir
# with the correct version.
_SPM_DEFAULTS['spm_dir_env_template'] = ['SPM{VERSION_NB}_DIR', 'SPM_DIR']

_SPM_DEFAULTS['matlab_exec_env'] = ['MATLAB_EXEC']


def _unique(seq):
    existing = set()
    ex_add = existing.add
    return [i for i in seq if not(i in existing or ex_add(i))]


def _get_defaults(template_name, templates_dict, version_nb=None):
    """get paths from a dict of path templates and a version number

    Parameters
    ----------
    template_name: str
    key which templates_dict maps to a list of templates to be used.

    templates_dict: dict
    dictionary of path templates to complete with version_nb.
    should map template_name to a sequence of strings (supposed to represent
    paths in the filesystem). Wherever the substring "{VERSION_NB}" appears in
    these strings, it will be replaced with version_nb.

    version_nb: int or str, optional (default=None)
    version number with which to complete the templates.
    If None and "version_numbers" is a key in templates_dict,
    each of the elements in templates_dict["version_numbers"] is used
    in turn, yielding len(templates_dict["version_numbers"]) results for
    each template.
    if version_nb is None and "version_numbers" is not a key in templates_dict,
    only "" (the empty string) is used to complete templates.

    Returns
    -------
    all_paths: list of str
    all the templates in templates_dict[template_name], formatted with
    version_nb. duplicate values are removed.

    """
    if templates_dict.get(template_name) is None:
        return []
    if version_nb is None:
        version_nb = templates_dict.get('version_numbers', [''])
    else:
        version_nb = [version_nb]
    all_paths = [template.format(VERSION_NB=nb) for
                 nb in version_nb for
                 template in templates_dict[template_name]]
    return _unique(all_paths)


def _get_exported(template_name, templates_dict, version_nb=None):
    env_vars = _get_defaults(template_name, templates_dict, version_nb)
    return [os.environ.get(var) for var in env_vars]


def _check_nipype_version():
    if distutils.version.LooseVersion(
            nipype.__version__).version < [0, 9]:
        return False
    return True


def _find_or_warn(loc, check, msg=None,
                  warner=_logger.warning, recursive=False):
    """ look for a location which verifies a certain condition ('check').

    Parameters
    ----------
    loc: str
    path to file or directory we want to validate.

    check: callable
    used to validate the path, must return True if
    it satisfies the condition and False otherwise.

    msg: str, optional (default=None)
    message to log if validation fails (if check returns False)
    if None nothing is logged.

    warner: callable, optional (default=configure_spm._logger.warning)
    called with warner(msg) if validation fails.

    recursive: bool, optional (default=False)
    if True and loc is the path to a directory, look for a path
    that verifies check recursively inside loc if check(loc) is False.

    Returns
    -------
    found:
    absolute path to the location which verifies check, if one was found.
    None:
    if no path was found which verifies check.

    """
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


def _find_or_warn_in_seq(loc_seq, check, msg=None,
                         warner=_logger.warning, recursive=False):
    """call _find_or_warn for each element of a sequence"""
    for loc in loc_seq:
        message = None if msg is None else msg.format(loc)
        found = _find_or_warn(
            loc, check, msg=message, warner=warner, recursive=recursive)
        if found is not None:
            return found

    return None


def _find_dep_loc(cli_loc=None, config_loc=None,
                  exported_locs=None, default_locs=None,
                  check=os.path.exists, msg_prefix='', recursive=False):
    """look for a file or directory in the filesystem.

    Parameters:
    ----------
    cli_loc: str, optional (default=None)
    first path we try to validate (if it is not None).
    (supposed to have been passed from the command line)

    config_loc: str, optional (default=None)
    first path we try, after cli_loc (if it is not None).
    (supposed to have been read in a config file)

    exported_locs: iterable containing str elements, optional (default=None)
    paths we try, in the order in which they are provided, if cli_loc
    and config_loc fail. (supposed to be paths read from the relevant
    environment variables).

    default_locs: iterable containing str elements, optional (default=None)
    paths we try, in the order in which they are provided, if all else
    fails. (supposed to be default, hard-coded locations).

    check: callable, optional (default=os.path.exists)
    checks the condition the path we are looking for must verify.
    returns True if a path verifies the condition and False otherwise.

    msg_prefix: str, optional (default='')
    message which is prepended to all logged messages, can be used to
    indicate what we are searching the filesystem for.

    recursive: bool, optional (default=False)
    also look recursively in the contents if the tested paths if they
    are directories.

    Returns
    -------
    found: str
    absolute path to the first location that was tried and for which
    check returned True.
    None:
    if no such location was found.

    """
    message = '{} cli specified location: "{}"'.format(msg_prefix, cli_loc)
    found = _find_or_warn(cli_loc, check, msg=message, recursive=recursive)
    if found is not None:
        return found

    message = '{} config file specified location: "{}"'.format(
        msg_prefix, config_loc)
    found = _find_or_warn(config_loc, check, msg=message, recursive=recursive)
    if found is not None:
        return found

    message = ' '.join([msg_prefix, 'exported location: "{}"'])
    found = _find_or_warn_in_seq(
        exported_locs, check, msg=message, recursive=recursive)
    if found is not None:
        return found

    message = ' '.join([msg_prefix, 'default location: "{}"'])
    found = _find_or_warn_in_seq(default_locs, check, msg=message,
                                 recursive=recursive, warner=_logger.debug)
    if found is not None:
        return found

    return None


def _is_executable(file_name):
    return (os.path.isfile(file_name) and os.access(file_name, os.X_OK))


def _guess_spm_version(spm_path, prefix_msg=''):
    """try to guess the spm version from numbers seen in the given path."""
    numbers = re.findall(r'[Ss][Pp][Mm][_\-\.]?(\d+)', spm_path)
    if not numbers:
        _logger.warning(
            '{} could not figure out SPM version number from path: "{}"'.format(
                prefix_msg, spm_path))
        return None

    numbers_set = set((int(number) for number in numbers))
    if len(numbers_set) == 1:
        return numbers_set.pop()

    best_guess = int(numbers[-1])
    if not _ACCEPT_SPM_MCR_WITH_AMBIGUOUS_VERSION:
        _logger.warning('{}: _ACCEPT_SPM_MCR_WITH_AMBIGUOUS_VERSION is '
                        'cleared: rejecting "{}"'.format(prefix_msg, spm_path))
        return None

    _logger.warning(
        '{} spm version number is ambiguous from spm path: "{}"; '
        'choosing number which appears last in the path: {}'.format(
            prefix_msg, spm_path, best_guess))
    return best_guess


def _is_spm_dir(dir_path, mcr_version=None):
    """check if dir_path seems to be an SPM home directory."""
    # a heuristic to know we have found the home for SPM is that
    # it contains a directory named tpm.
    if not os.path.isdir(os.path.join(dir_path, 'tpm')):
        return False
    if mcr_version is None:
        return True
    dir_version = _guess_spm_version(dir_path, 'looking for spm dir version:')
    if dir_version == mcr_version:
        return True
    _logger.debug(
        'spm versions mismatch: mcr version: {}; dir version: {}; '
        'for candidate spm dir "{}"'.format(
            mcr_version, dir_version, dir_path))
    return False


class _IsValidMCR(object):
    """used to validate path to SPM MCR and capture path to matching SPM dir.

    meant to be used as the 'check' argument of _find_dep_loc.
    """

    def __init__(self, cli_spm_dir, config_spm_dir, defaults):
        """
        specify the locations in which a matching SPM dir will be seeked.

        Parameters
        ----------
        cli_spm_dir: str
        first location tried when looking for SPM dir.
        (supposed to have been specified from the command line)

        config_spm_dir: str
        next location to be tried.
        (supposed to have been specified in a config file)

        defaults: dict
        mapping of relevant environment variables and default locations
        in which to look if cli and config fail. Must provide the same semantics
        as configure_spm._SPM_DEFAULTS. (but not necessarily every key present
        in it: if some of the keys provided in _SPM_DEFAULTS are missing from
        defaults, they will simply not be used). see documentation for
        _get_defaults for more details.

        Returns
        -------
        None

        """
        self.cli_spm_dir_ = cli_spm_dir
        self.config_spm_dir_ = config_spm_dir
        self.defaults_ = defaults
        self.found_spm_dir_ = None
        self.spm_version_ = None

    def __call__(self, spm_mcr):
        """check that path points to SPM MCR and a matching SPM dir exists

        first, check that path points to an executable file and if it does,
        look for an SPM directory with matching version. If one is found,
        remember it in self.found_spm_dir_.

        Parameters
        ----------
        spm_mcr: path to (supposed) SPM MCR to be examined.

        Returns
        -------
        True
        if path points to an MCR and an SPM home directory with
        matching version number was found (it can then be looked up
        in self.found_spm_dir_).
        False
        otherwise.

        """
        if not _is_executable(spm_mcr):
            return False
        _logger.debug('found SPM MCR in "{}"'.format(spm_mcr))
        mcr_version = _guess_spm_version(
            spm_mcr, 'looking for spm mcr version:')
        if not _ACCEPT_SPM_MCR_WITH_UNKNOWN_VERSION and mcr_version is None:
            _logger.warning('_ACCEPT_SPM_MCR_WITH_UNKNOWN_VERSION is '
                            'cleared: rejecting "{}"'.format(spm_mcr))
            return False
        spm_dir_envs = _get_exported(
            'spm_dir_env_template', self.defaults_, version_nb=mcr_version)
        spm_dir_defaults = _get_defaults(
            'spm_dir_template', self.defaults_, version_nb=mcr_version)

        # we found the executable; add its parent directory
        # to default locations in which to look for SPM home.
        spm_dir_defaults.insert(0, os.path.dirname(spm_mcr))
        spm_dir = _find_dep_loc(
            self.cli_spm_dir_, self.config_spm_dir_,
            spm_dir_envs, spm_dir_defaults,
            check=partial(_is_spm_dir, mcr_version=mcr_version),
            msg_prefix='SPM directory could not be found in',
            recursive=True)
        if spm_dir is None:
            _logger.debug(
                'could not find matching SPM dir for mcr: "{}"'.format(spm_mcr))
            return False
        self.found_spm_dir_ = spm_dir
        self.spm_version_ = mcr_version
        _logger.info('found SPM directory to be "{}"'.format(spm_dir))
        return True


def _find_spm_mcr_and_spm_dir(cli_spm_mcr, config_spm_mcr,
                              cli_spm_dir, config_spm_dir, defaults):
    """look for an SPM MCR and an SPM home dir with matching version numbers.

    We also check that the nipype version is recent enough for MCR
    to be used.

    Parameters
    ----------
    cli_spm_mcr: str
    first location to try for mcr. (supposed to come from command line)

    config_spm_mcr: str
    second location to try for mcr. (supposed to come from config file)

    cli_spm_dir: str
    first location to try for SPM home dir. (supposed to come from command line)

    config_spm_dir: str
    second location to try for SPM home dir. (supposed to come from config file)

    defaults_ dict
    mapping of relevant environment variables and default locations
    in which to look if cli and config fail. Must provide the same semantics
    as configure_spm._SPM_DEFAULTS. (but not necessarily every key present
    in it: if some of the keys provided in _SPM_DEFAULTS are missing from
    defaults, they will simply not be used). see documentation for
    _get_defaults for more details.

    Returns
    -------
    spm dir: str
    path to the SPM home directory that was found if checks succeeded.
    None
    otherwise.

    """
    if not _check_nipype_version():
        _logger.warning(
            'nipype version {} too old.'
            ' No support for precompiled SPM'.format(nipype.__version__))
        return None

    check_mcr = _IsValidMCR(cli_spm_dir, config_spm_dir, defaults)
    spm_mcr_envs = _get_exported('spm_mcr_env_template',
                                 templates_dict=defaults)
    spm_mcr_defaults = _get_defaults('spm_mcr_template',
                                     templates_dict=defaults)
    spm_mcr = _find_dep_loc(
        cli_spm_mcr, config_spm_mcr, spm_mcr_envs, spm_mcr_defaults,
        check=check_mcr,
        msg_prefix='SPM MCR has no matching dir or could not be found in')
    if spm_mcr is None:
        _logger.warning('failed to find SPM MCR or SPM directory')
        return None

    spm_dir = check_mcr.found_spm_dir_
    spm_version = check_mcr.spm_version_

    return spm_mcr, spm_dir, spm_version


def _configure_spm_using_mcr(spm_mcr, spm_dir, spm_version):
    """configure SPM backend given paths to SPM MCR, home dir and version."""
    if spm_version is not None:
        _logger.info('using SPM version: {}'.format(spm_version))
    _logger.info('setting SPM MCR path to "{}" '
                 'and "use_mcr" to True'.format(spm_mcr))
    if spm_version >= 12:
        spm.SPMCommand.set_mlab_paths(
            matlab_cmd='{} batch'.format(spm_mcr), use_mcr=True)
    else:
        spm.SPMCommand.set_mlab_paths(
            matlab_cmd='{} run script'.format(spm_mcr), use_mcr=True)
    _logger.info('SPM configuration succeeded using SPM MCR.')


def _find_matlab_exec_and_spm_dir(
        cli_matlab_exec, config_matlab_exec,
        cli_spm_dir, config_spm_dir, defaults):
    """look for a Matlab executable and an SPM home directory.

    Parameters
    ----------
    cli_matlab_exec: str
    first location to try for Matlab. (supposed to come from command line)

    config_matlab_exec: str
    second location to try for Matlab. (supposed to come from config file)

    cli_spm_dir: str
    first location to try for SPM home dir. (supposed to come from command line)

    config_spm_dir: str
    second location to try for SPM home dir. (supposed to come from config file)

    defaults_ dict
    mapping of relevant environment variables and default locations
    in which to look if cli and config fail. Must provide the same semantics
    as configure_spm._SPM_DEFAULTS. (but not necessarily every key present
    in it: if some of the keys provided in _SPM_DEFAULTS are missing from
    defaults, they will simply not be used). see documentation for
    _get_defaults for more details.

    Returns
    -------
    spm dir: str
    path to the SPM home directory that was found if checks succeeded.
    None
    otherwise.

    """
    # if using licensed matlab we need both matlab executable and
    # the home spm directory.
    matlab_envs = _get_exported('matlab_exec_env', templates_dict=defaults)
    matlab_defaults = _get_defaults('matlab_exec', templates_dict=defaults)
    matlab_exec = _find_dep_loc(
        cli_matlab_exec, config_matlab_exec,
        matlab_envs, matlab_defaults,
        check=_is_executable,
        msg_prefix='matlab executable file could not be found in')
    if matlab_exec is None:
        _logger.warning('failed to find Matlab executable file')
        return None

    _logger.info('found matlab executable in: "{}"'.format(matlab_exec))
    spm_dir_envs = _get_exported('spm_dir_env_template',
                                 templates_dict=defaults)
    spm_dir_defaults = _get_defaults('spm_dir_template',
                                     templates_dict=defaults)
    spm_dir = _find_dep_loc(
        cli_spm_dir, config_spm_dir, spm_dir_envs, spm_dir_defaults,
        check=_is_spm_dir,
        msg_prefix='SPM directory could not be found in',
        recursive=True)

    if spm_dir is not None:
        _logger.info('found SPM directory to be "{}"'.format(spm_dir))
        spm_version = _guess_spm_version(spm_dir)
        return matlab_exec, spm_dir, spm_version

    _logger.warning('failed to find SPM directory')
    return None


def _configure_spm_using_matlab(matlab_exec, spm_dir, spm_version):
    """config SPM backend given paths to Matlab and SPM home dir and version."""
    if spm_version is not None:
        _logger.info('using SPM version: {}'.format(spm_version))
    _logger.info('setting default matlab command to be "{}"'.format(
        matlab_exec))
    matlab.MatlabCommand.set_default_matlab_cmd(matlab_exec)
    _logger.info('setting matlab default paths to "{}"'.format(spm_dir))
    matlab.MatlabCommand.set_default_paths(spm_dir)
    _logger.info('SPM configuration succeeded using Matlab.')


def _configure_spm(cli_spm_dir=None, config_spm_dir=None,
                   cli_matlab_exec=None, config_matlab_exec=None,
                   cli_spm_mcr=None, config_spm_mcr=None, defaults=None,
                   prefer_matlab=False):
    """configure the SPM backend and return path to SPM home directory.

    We first try to find an SPM precompiled MCR and an SPM home directory
    with matching version number, and if this fails we look for a
    Matlab executable and any SPM home directory.

    Parameters
    ----------
    cli_spm_dir: str, optional (default=None)
    first location to try for SPM home dir. (supposed to come from command line)

    config_spm_mcr: str, optional (default=None)
    second location to try for SPM home dir. (supposed to come from config file)

    cli_matlab_exec: str, optional (default=None)
    first location to try for Matlab. (supposed to come from command line)

    config_matlab_exec: str, optional (default=None)
    second location to try for Matlab. (supposed to come from config file)

    cli_spm_mcr: str, optional (default=None)
    first location to try for mcr. (supposed to come from command line)

    config_spm_mcr: str, optional (default=None)
    second location to try for mcr. (supposed to come from config file)

    defaults: dict, optional (default=None)
    if None configure_spm._SPM_DEFAULTS is used.
    mapping of relevant environment variables and default locations
    in which to look if cli and config fail. Must provide the same semantics
    as configure_spm._SPM_DEFAULTS. (but not necessarily every key present
    in it: if some of the keys provided in _SPM_DEFAULTS are missing from
    defaults, they will simply not be used). see documentation for
    _get_defaults for more details.

    prefer_matlab: bool, optional (default=False)
    if True try configuring SPM using Matlab first and try to use SPM MCR only
    if this fails; if False start with SPM MCR.

    Returns
    -------
    found_spm_dir: str
    path to SPM home directory if we found either:
    - a viable SPM MCR and an SPM home directory with matching version
    - or a viable Matlab executable and an SPM home directory
    None
    if SPM configuration failed
    """
    if defaults is None:
        defaults = _SPM_DEFAULTS
    find_with_mcr = partial(_find_spm_mcr_and_spm_dir,
                            cli_spm_mcr, config_spm_mcr,
                            cli_spm_dir, config_spm_dir,
                            defaults)
    find_with_matlab = partial(_find_matlab_exec_and_spm_dir,
                               cli_matlab_exec, config_matlab_exec,
                               cli_spm_dir, config_spm_dir,
                               defaults)

    first, second = ({'find': find_with_mcr,
                      'configure': _configure_spm_using_mcr},
                     {'find': find_with_matlab,
                      'configure': _configure_spm_using_matlab})
    if prefer_matlab:
        first, second = second, first
    for choice in (first, second):
        found = choice['find']()
        if found is not None:
            executable, spm_dir, spm_version = found
            choice['configure'](*found)
            return spm_dir

    _logger.error(
        'could not find a pair (SPM_MCR precompiled executable, SPM directory)'
        ' nor a pair (Matlab executable, SPM directory): '
        'SPM configuration failed.')
    return None


def _get_version_spm(spm_dir):
    """used by preproc_reporter; for other uses prefer _guess_spm_version"""
    # Return current SPM version
    _, spm_version = os.path.split(spm_dir)
    if not spm_version:
        _logger.warning('Could not figure out SPM version!'
                        ' (spm dir: "{}")'.format(spm_dir))
        return
    return spm_version
