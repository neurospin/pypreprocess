import spm_loader
from distutils.core import setup

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix

"""

setup(
    name='spm_loader',
    description='Utility functions to extract relevant information from an SPM.mat file',
    # long_description=open('README.md').read(),
    version=spm_loader.__version__,
    author='Yannick Schwartz',
    packages = ['spm_loader'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
)
