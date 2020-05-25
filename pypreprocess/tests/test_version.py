"""
A simple test script for existance of pypreprocess version
"""

import pypreprocess
from ..version import __version__


def test_version():
    version = pypreprocess.__version__
    assert isinstance(version, str)
