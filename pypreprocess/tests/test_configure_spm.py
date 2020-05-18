# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:01:12 2015

@author: Mehdi RAHIM, Jerome DOCKES
"""
import os
import stat
import json
import nipype
import pytest

from distutils.version import LooseVersion
from pypreprocess.configure_spm import (
    _get_version_spm, _logger, _configure_spm,
    _guess_spm_version, _ACCEPT_SPM_MCR_WITH_AMBIGUOUS_VERSION)


def test_get_version_spm():
    spm_version_1 = _get_version_spm('/tmp/path/to/spm8')
    spm_version_2 = _get_version_spm('/path/to/spm12')
    assert spm_version_1 == 'spm8'
    assert spm_version_2 == 'spm12'


def test_guess_spm_version():
    spm_version = _guess_spm_version('/tmp/some23/Spm12/SPM12')
    assert spm_version == 12
    spm_version = _guess_spm_version('/tmp/1/spm_8.sh')
    assert spm_version == 8
    spm_version = _guess_spm_version('spm_using_mlab_8/spm/')
    assert spm_version is None
    spm_version = _guess_spm_version('spm/8/spm/')
    assert spm_version is None
    spm_version = _guess_spm_version('/home/sp12/spm/spm.sh')
    assert spm_version is None
    spm_version = _guess_spm_version('/opt/spm8/spm12.sh')
    if(_ACCEPT_SPM_MCR_WITH_AMBIGUOUS_VERSION):
        assert spm_version == 12
    else:
        assert spm_version is None


def _make_dirs(root, body):
    """create an arborescence rooted at root specified by body

    used to create a dummy arborescence to test functions which
    search the filesystem for dependencies, e.g. an SPM installation.

    Parameters
    ----------
    root: str
    path to the directory (which does not necessarily exist) which
    will be the top directory of the created arborescence.

    body: dict
    specify the contents of the created arborescence.
    it is a dictionary of {'dirname': contents} pairs,
    where contents is of the form:
    {
    'dirs': {'subdir_0': contents_0, 'sudir_1': contents_1, ...},
    'files': [filename_0, filename_1, ...]
    }
    where contents_0, contents_1, ... have the same form as contents.
    see _get_spm_body._spm_body_template_dict for an example

    Returns
    -------
    None

    """
    root = os.path.abspath(os.path.expanduser(root))
    try:
        os.makedirs(root)
    # except FileExistsError as e:
    #     pass
    # with python2 there is no FileExistsError
    except OSError as e:
        if e.errno != 17: # 17 is FileExists
            raise(e)
    sub_dirs = body.get('dirs', {})
    for sub_dir, sub_body in sub_dirs.items():
        _make_dirs(os.path.join(root, sub_dir), sub_body)
    for reg_file in body.get('files', []):
        with open(os.path.join(root, reg_file), 'w') as f:
            pass


def _get_spm_body(version_nb):
    """get a body for _make_dirs, representing an SPM install.

    Parameters
    ----------
    version_nb: int
    version number with which directory and file names will
    be formatted.

    Returns
    -------
    spm_body: dict
    spm installation-like arborescence

    """
    spm_body = json.loads(
        _get_spm_body._spm_body_template_json % {'VERSION_NB': version_nb})
    return spm_body


_get_spm_body._spm_body_template_dict = {
    "spm%(VERSION_NB)d": {
        "files": ["spm%(VERSION_NB)d.sh"],
        "dirs": {
            "mcr": {
                "dirs": {"_jvm": {}},
                "files": []
            },
            "spm%(VERSION_NB)d": {
                "files": ["run_spm%(VERSION_NB)d.sh"],
                "dirs": {
                    "spm%(VERSION_NB)d_mcr": {
                        "files": [],
                        "dirs": {
                            "home": {},
                            "toolbox": {},
                            "spm%(VERSION_NB)d": {
                                "files": ["spm_ROI.m"],
                                "dirs": {
                                    "tpm": {},
                                    "templates": {},
                                    "toolbox": {},
                                    "src": {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

_get_spm_body._spm_body_template_json = json.dumps(
    _get_spm_body._spm_body_template_dict)


def _spm_test_dummy_arborescence(root='/tmp/', spm_versions=[7, 8, 12, 13]):
    """return an arborescence simulating several SPM versions and Matlab

    Parameters
    ----------
    root: str, optional (default='/tmp/')
    directory in which we will want to create the dummy arborescence.

    spm_versions: list, optional (default=[7, 8, 12, 13])
    spm versions (different simulated installations) to include.

    Returns
    -------
    spm_root, body: str, dict
    spm_root is root/spm_dummy.
    body is a representation of the arborescence to be
    passed to _make_dirs in order to create the test directory.

    """
    root = os.path.abspath(os.path.expanduser(root))
    spm_root = os.path.join(root, 'spm_dummy')

    body = {
        'dirs': {'matlab': {'files': ['matlab.sh']}},
        'files': []
    }

    for version in spm_versions:
        body['dirs'].update(_get_spm_body(version))

    return spm_root, body


def _get_test_spm_starting_defaults(spm_root='/tmp/spm_dummy/'):
    """get default paths and env variable names used to test _configure_spm."""
    start_defaults = {}

    start_defaults['version_numbers'] = [13, 12, 8]

    start_defaults['spm_mcr_template'] = [
        os.path.join(spm_root,
                     'bad_version{VERSION_NB}_specific_mcr_loc.sh.BAD_LOC'),
        os.path.join(spm_root,
                     'bad_version_independant_mcr_loc.sh.BAD_LOC'),
        os.path.join(spm_root,
                     'spm{VERSION_NB}/spm{VERSION_NB}.sh.BAD_LOC')]

    start_defaults['spm_dir_template'] = [
        os.path.join(spm_root,
                     'bad_version{VERSION_NB}_specific_spm_dir_loc.BAD_LOC'),
        os.path.join(spm_root,
                     'bad_version_independant_specific_spm_dir_loc.BAD_LOC'),
        os.path.join(spm_root,
                     'spm{VERSION_NB}.BAD_LOC')]

    start_defaults['matlab_exec'] = [
        os.path.join(spm_root,
                     'bad_matlab_location.BAD_LOC'),
        os.path.join(spm_root,
                     'matlab/matlab.sh.BAD_LOC')]

    start_defaults['spm_mcr_env_template'] = ['TEST_SPM_MCR',
                                              'TEST_SPM{VERSION_NB}_MCR']

    start_defaults['spm_dir_env_template'] = ['TEST_SPM{VERSION_NB}_DIR',
                                              'TEST_SPM_DIR']

    start_defaults['matlab_exec_env'] = ['TEST_SPM_MATLAB_EXEC']

    return start_defaults


def _fix_spm_testing_default(defaults_dict, key_to_fix):
    _logger.debug('SPM config testing: fixing {}'.format(key_to_fix))
    defaults_dict[key_to_fix] = [
        s.replace('.BAD_LOC', '') for s in defaults_dict[key_to_fix]]


def _break_spm_testing_default(defaults_dict, key_to_fix):
    _logger.debug('SPM config testing: breaking {}'.format(key_to_fix))
    defaults_dict[key_to_fix] = ['.'.join([s, 'BAD_LOC']) for
                                 s in defaults_dict[key_to_fix]]


def _fix_spm_testing_explicitly_set_location(locations_dict, key_to_fix):
    _logger.debug('SPM config testing: fixing {}'.format(key_to_fix))
    locations_dict[key_to_fix] = locations_dict[
        key_to_fix].replace('.BAD_LOC', '')


def _execute_spm_config_test(defaults, explicitly_set, spm_root):
    """execute tests prepared by test_spm_config."""
    defaults = dict(defaults)
    explicitly_set = dict(explicitly_set)

    # at the beginning all paths point to bad locations,
    # configuration should fail
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert spm_dir is None
    _fix_spm_testing_default(defaults, 'matlab_exec')
    # path to Matlab was fixed but still no path to SPM home dir,
    # configuration should fail.
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert spm_dir is None
    _fix_spm_testing_default(defaults, 'spm_dir_template')
    # path to Matlab is good, path to spm13 is valid but this installation
    # is broken (missing tpm dir), so spm12 should be selected.
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert os.path.samefile(spm_dir,
            os.path.join(spm_root, 'spm12/spm12/spm12_mcr/spm12/'))
    _break_spm_testing_default(defaults, 'spm_dir_template')
    _fix_spm_testing_default(defaults, 'spm_mcr_template')
    # path to Matlab is bad but paths to SPM MCRs are good,
    # paths to SPM home dirs are bad but can be inferred from
    # SPM MCR location; configuration should succeed and select
    # spm12 (spm13 is broken).
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert os.path.samefile(spm_dir,
            os.path.join(spm_root, 'spm12/spm12/spm12_mcr/spm12/'))
    _logger.debug('setting TEST_SPM_MCR env variable')
    os.environ['TEST_SPM_MCR'] = os.path.join(spm_root, 'spm7/spm7.sh/')
    # environment variables have priority over defaults, so spm7
    # should be preferred.
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert os.path.samefile(spm_dir,
            os.path.join(spm_root, 'spm7/spm7/spm7_mcr/spm7/'))
    _fix_spm_testing_explicitly_set_location(explicitly_set, 'config_spm_mcr')
    # path to spm specified in config file was fixed and points to
    # spm8, spm8 should be chosen.
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert os.path.samefile(spm_dir,
            os.path.join(spm_root, 'spm8/spm8/spm8_mcr/spm8/'))
    _fix_spm_testing_explicitly_set_location(explicitly_set, 'cli_spm_mcr')
    # path to spm specified in command line was fixed and points to spm7,
    # since cli has priority over config file, spm7 should be chosen.
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert os.path.samefile(spm_dir,
            os.path.join(spm_root, 'spm7/spm7/spm7_mcr/spm7/'))
    _logger.debug('setting "prefer_matlab"')
    _fix_spm_testing_default(defaults, 'spm_dir_template')
    # paths to SPM homes were fixed; since we prefer Matlab the version
    # of the preferred MCR has no importance and is ignored; the highest
    # valid install (spm12) should be chosen.
    spm_dir = _configure_spm(defaults=defaults, prefer_matlab=True,
                             **explicitly_set)
    assert os.path.samefile(spm_dir,
            os.path.join(spm_root, 'spm12/spm12/spm12_mcr/spm12/'))

    _logger.info('SPM config test succeeded')

@pytest.mark.skipif(LooseVersion(nipype.__version__) < LooseVersion('1.1.3'),
                    reason='NiPype ver < 1.1.3 has a bug due to which'
                           'it fails to catch a legitimate exception, '
                           'causing spurious test failures',
                    )

@pytest.mark.skip()
def test_spm_config(scratch_dir='/tmp/'):
    """prepare dir containing fake Matlab and SPM installs and launch tests."""
    scratch_dir = os.path.abspath(os.path.expanduser(scratch_dir))
    spm_root, dir_spec = _spm_test_dummy_arborescence(root=scratch_dir)

    # break spm13 installation
    dir_spec['dirs']['spm13']['dirs']['spm13']['dirs']['spm13_mcr']['dirs'] = {}

    _make_dirs(spm_root, dir_spec)

    for v_nb in [7, 8, 12, 13]:
        spm_exec_path = os.path.join(
            spm_root, 'spm{nb}/spm{nb}.sh'.format(nb=v_nb))
        st = os.stat(spm_exec_path)
        os.chmod(spm_exec_path, st.st_mode | stat.S_IEXEC)

    matlab_exec_path = os.path.join(spm_root, 'matlab/matlab.sh')
    st = os.stat(matlab_exec_path)
    os.chmod(matlab_exec_path, st.st_mode | stat.S_IXUSR)

    defaults = _get_test_spm_starting_defaults(spm_root=spm_root)
    explicitly_set = {'cli_spm_dir': os.path.join(spm_root, 'spm7.BAD_LOC'),
                      'config_spm_dir': os.path.join(spm_root, 'spm8.BAD_LOC'),
                      'cli_matlab_exec': os.path.join(
                          spm_root, 'matlab/matlab.sh.BAD_LOC'),
                      'config_matlab_exec': os.path.join(
                          spm_root, 'matlab/matlab_config.sh.BAD_LOC'),
                      'cli_spm_mcr': os.path.join(
                          spm_root, 'spm7/spm7.sh.BAD_LOC'),
                      'config_spm_mcr': os.path.join(
                          spm_root, 'spm8/spm8.sh.BAD_LOC')}
    _execute_spm_config_test(defaults, explicitly_set, spm_root)


if __name__ == '__main__':
    test_get_version_spm()
    test_guess_spm_version()
    test_spm_config()
