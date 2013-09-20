from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

sourcefiles = ['spm_hist2py.pyx', 'spm_hist2.c']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("spm_hist2py", sourcefiles)]
)

