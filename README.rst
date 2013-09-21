.. -*- mode: rst -*-

pypreprocess
============
**pypreprocess** is a collection of python scripts and modules for preprocessing (motion 
correction, spatial normalization, smoothing, etc.) fMRI data. It also contains utilities for automatic 
QA like registration checks (using nipy.labs), and template-based html report
generation using (tempita, jquery, and home-grown css). It contains:
  * utility scripts modules and for wrapping SPM and FSL (via nipype), and also
  * utilitiy scripts and modules for doing preprocessing in pure-python (for minimal single-subject preprocessing --sclice timing correction + motion correction + coregistration + smoothing-- in pure python)


Progressively, it contains the preprocessing is been done using pure-pythin code (void of nipype, etc.).
See for example, the script ``examples/purepython_preproc_demo.py``, an example script for minimal single-subject
preprocessing (clice timing correction + motion correction + coregistration + smoothing) in pure python

This work is made available by the INRIA Parietal Project.

Important links
===============

- Official repo: https://github.com/neurospin/pypreprocess

Dependencies
============
  * Python >= 2.6
  * Numpy >= 1.3
  * SciPy >= 0.7
  * nipype >= 0.8.0
  * nipy >= 0.3.0
  * traits >= 4.3.0
  * joblib >= 0.7.0
  * nibabel >= 1.3.0
  * networkx >= 1.7
  * sympy >= 0.7.1
  * matplotlib >= 0.99.1


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install


Use-case Examples
=================
We have written some examplary scripts for preprocessing some popular datasets.


SPM auditory single-subject
-----------------------------
cd to the ``examples/`` directory, and run the following command:

       python nipy_glm_spm_auditory.py spm_auditory spm_auditory_runs/ 

Now open the file spm_auditory_runs/subXYZ/report.html in your browser (firefox), to see
the generate report (QA).

Production scripts
------------------
The ``scripts/`` sub-folder contains scripts for preprocessing popular datasets like ABIDE, HCP, HAXBY2001, NYU rest, etc.
For example the script ``scripts/abide_preproc.py`` preprocesses the ABIDE data.

Testing
=======
  nosetools -v pypreprocess/tests
