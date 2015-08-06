# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:01:12 2015

@author: Mehdi RAHIM
"""

from nose.tools import assert_equal
from pypreprocess.configure_spm import _get_version_spm

def test_get_version_spm():
    spm_version_1 = _get_version_spm('/tmp/path/to/spm8')
    spm_version_2 = _get_version_spm('/path/to/spm12')
    assert_equal(spm_version_1, 'spm8')
    assert_equal(spm_version_2, 'spm12')
