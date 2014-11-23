pypreprocess
============
**pypreprocess** is a collection of python scripts for preprocessing (motion 
correction, spatial normalization, smoothing, etc.) fMRI data using 
nipype's SPM and FSL interfaces. It also contains utilities for automatic 
QA like registration checks (using nipy.labs), and template-based html report
generation using (tempita, jquery, and home-grown css).

These days, it also contains pure-Python (no C extensions, no compiled code, just Python)
modules and scripts for slice-timing correction, motion correction, coregistration,
and smoothing, without need for nipype or matlab.

Check out the wiki at https://github.com/neurospin/pypreprocess/wiki.

This work is made available by the Parietal https://team.inria.fr/parietal/.


CI status
=========
[![Build Status](https://travis-ci.org/dohmatob/pypreprocess.svg?branch=master)](https://travis-ci.org/dohmatob/pypreprocess)


Important links
===============

- Official source code repo: https://github.com/neurospin/pypreprocess

Dependencies
============
        * Python >= 2.6
        * Numpy >= 1.3
        * SciPy >= 0.7
        * matplotlib >= 0.99.1
        * nibabel >= 1.3.0
        * networkx >= 1.7
        * traits >= 4.3.0
        * sympy >= 0.7.4.1
        * nipype >= 0.8.0
        * nipy >= 0.3.0	
	* configobj >= 5.0.6


Installation
============
Ensure that you have the above dependencies installed. Then run:

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install


Use-case Examples
=================
We have written some examplary scripts for preprocessing some popular datasets.

SPM auditory (single-subject)
-----------------------------
cd to the pypreprocess/examples directory, and run the following command:

       python nipype_preproc_spm_auditory.py 

Now open the file spm_auditory_runs/sub001/report.html in your browser (firefox), to see
the generate report (QA).

'Serious' examples
------------------
The scripts/ sub-folder contains scripts for preprocessing popular datasets like ABIDE, HCP, HAXBY2001, NYU rest, etc.
They should work 'out-of-the-box'.


Intra-subject preprocessing in pure-Python
==========================================
cd to the pypreprocess/examples directory, and run the following command:

       python purepython_preproc_demo.py
       

Using .ini configuration files to specify pipeline
==================================================
Take a look at the examples/ subdirectory for use-cases of this feature.


Development
===========

Code
----

GIT
~~~
You can check the latest sources with the command::

    git clone git://github.com/neurospin/pypreprocess.git

or if you have write privileges::

    git clone git@github.com:neurospin/pypreprocess.git
