.. -*- mode: rst -*-

.. image:: https://travis-ci.org/neurospin/pypreprocess.svg?branch=master
   :target: https://travis-ci.org/neurospin/pypreprocess
   :alt: Build Status
   
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
* nilearn >= 0.1.1


Installation
============
First install the above dependencies by running the following commands in a terminal:

       wget -O- http://neuro.debian.net/lists/precise.us-nh.libre | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
       sudo apt-key adv --recv-keys --keyserver pgp.mit.edu 2649A5A9
       sudo apt-get update
       sudo apt-get install python-scipy python-nose python-nibabel python-sklearn python-sympy python-networkx python-nipy python-nipype
       sudo apt-get install python-pip && pip install nilearn --user


Then install pypreprocess itself by running:

       python setup.py install --user


Use-case Examples
=================
We have written some examplary scripts for preprocessing some popular datasets. More information about examples
can be found on the [wiki!](https://github.com/neurospin/pypreprocess/wiki/Example-scripts)

The **examples** directory contains a set of scripts, each demoing an aspect of pypreprocessing. Some scripts even provide use-cases for the nipy-based GLM. The examples use publicly available sMRI and fMRI data. Data fetchers are based on the nilearn API.

The main examples scripts can be summarized as follows:
* **examples/nipype_preproc_spm_auditory.py**: demos preprocessing + first-level GLM (using nipy)  on the single-subject SPM auditory dataset.
* **examples/nipype_preproc_spm_multimodal_faces.py**: demos preprocessing + first-level fixed-effects GLM on 
R. Henson's multi-modal face dataset (multiple sessions)
* **examples/nipy_glm_fsl_feeds_fmri.py**: demos preprocessing + first-level GLM on FSL FEEDS dataset
* **examples/slice_timing_demos.py, examples/realign_demos.py, examples/coreg_demos.py**: demos Slice-Timing Correction (STC), motion-correction, and coregistration on various datasets, using modules written in pure Python
* **examples/purepython_preproc_demo.py**: demos intra-subject preprocessing using pure Python modules, on single-subject SPM auditory dataset
* **examples/nipype_preproc_spm_nyu.py**: preprocessing of NYU resting-state dataset
* **examples/nipype_preproc_spm_haxby.py**: preprocessing of the 'Haxby2001' visual recognition task fMRI dataset.


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
It's possible (and recommended) to configure the preprocessing pipeline just by copying
any of the `.ini` configuration files under the `examples` sub-directory and modifying it accordingly (usually, you only need to modify the `dataset_dir` paramter).


Development
===========
You can check the latestC sources with the command::

       git clone git://github.com/neurospin/pypreprocess.git

or if you have write privileges::

       git clone git@github.com:neurospin/pypreprocess.git
