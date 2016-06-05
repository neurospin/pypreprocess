# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:01:12 2015

@author: Mehdi RAHIM
"""
import os
import stat
import json

from nose.tools import assert_equal
from pypreprocess.configure_spm import _get_version_spm, _logger, _configure_spm

def test_get_version_spm():
    spm_version_1 = _get_version_spm('/tmp/path/to/spm8')
    spm_version_2 = _get_version_spm('/path/to/spm12')
    assert_equal(spm_version_1, 'spm8')
    assert_equal(spm_version_2, 'spm12')

def _make_dirs(root, body):
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
    spm_body = json.loads(
        _get_spm_body._spm_body_template_json % {'VERSION_NB': version_nb})
    return spm_body

_get_spm_body._spm_body_template_dict = {
    "spm%(VERSION_NB)d": {
        "dirs": {
            "spm%(VERSION_NB)d": {
                "dirs": {
                    "spm%(VERSION_NB)d_mcr": {
                        "dirs": {
                            "home": {},
                            "toolbox": {},
                            "spm%(VERSION_NB)d": {
                                "dirs": {
                                    "tpm": {},
                                    "templates": {},
                                    "toolbox": {},
                                    "src": {}
                                },
                                "files": ["spm_ROI.m"]
                            }
                        },
                        "files": []
                    }
                },
                "files": ["run_spm%(VERSION_NB)d.sh"]
            },
            "mcr": {
                "dirs": {"_jvm": {}},
                "files": []
            }
        },
        "files": ["spm%(VERSION_NB)d.sh"]
    }
}

_get_spm_body._spm_body_template_json = json.dumps(
    _get_spm_body._spm_body_template_dict)

def _spm_test_dummy_arborescence(root='/tmp/', spm_versions=[7, 8, 12, 13]):

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

    start_defaults = {}

    start_defaults['spm_versions'] = [13, 12, 8]

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

    defaults = dict(defaults)
    explicitly_set = dict(explicitly_set)

    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert(spm_dir is None)
    _fix_spm_testing_default(defaults, 'matlab_exec')
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert(spm_dir is None)
    _fix_spm_testing_default(defaults, 'spm_dir_template')
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert(os.path.samefile(
        spm_dir, os.path.join(spm_root, 'spm12/spm12/spm12_mcr/spm12/')))
    _break_spm_testing_default(defaults, 'spm_dir_template')
    _fix_spm_testing_default(defaults, 'spm_mcr_template')
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert(os.path.samefile(
        spm_dir, os.path.join(spm_root, 'spm12/spm12/spm12_mcr/spm12/')))
    _logger.debug('setting TEST_SPM_MCR env variable')
    os.environ['TEST_SPM_MCR'] = os.path.join(spm_root, 'spm7/spm7.sh/')
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert(os.path.samefile(
        spm_dir, os.path.join(spm_root, 'spm7/spm7/spm7_mcr/spm7/')))
    _fix_spm_testing_explicitly_set_location(explicitly_set, 'config_spm_mcr')
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert(os.path.samefile(
        spm_dir, os.path.join(spm_root, 'spm8/spm8/spm8_mcr/spm8/')))
    _fix_spm_testing_explicitly_set_location(explicitly_set, 'cli_spm_mcr')
    spm_dir = _configure_spm(defaults=defaults, **explicitly_set)
    assert(os.path.samefile(
        spm_dir, os.path.join(spm_root, 'spm7/spm7/spm7_mcr/spm7/')))
    _logger.debug('setting "prefer_matlab"')
    spm_dir = _configure_spm(defaults=defaults, prefer_matlab=True,
                             **explicitly_set)
    assert(os.path.samefile(
        spm_dir, os.path.join(spm_root, 'spm7/spm7/spm7_mcr/spm7/')))
    _logger.debug('setting "prefer_matlab"')
    _fix_spm_testing_default(defaults, 'spm_dir_template')
    spm_dir = _configure_spm(defaults=defaults, prefer_matlab=True,
                             **explicitly_set)
    assert(os.path.samefile(
        spm_dir, os.path.join(spm_root, 'spm12/spm12/spm12_mcr/spm12/')))    

    _logger.info('SPM config test succeeded')

def test_spm_config(scratch_dir='/tmp/'):

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
    test_spm_config()
