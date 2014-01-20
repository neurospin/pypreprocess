# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
from distutils import log


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('nipy_labs', parent_package, top_path)
    config.add_subpackage('datasets')
    config.add_subpackage('viz_tools')
    config.add_subpackage('utils')
    config.make_config_py() # installs __config__.py
    return config

if __name__ == '__main__':
    #from numpy.distutils.core import setup
    #setup(**configuration().todict())
    print('This is the wrong setup.py file to run')
