"""
A simple test script for existance of pypreprocess version
"""

import pypreprocess

from nose.tools import assert_true
from ..version import __version__


def test_version():
    version = pypreprocess.__version__
    assert_true(isinstance(version, str))
