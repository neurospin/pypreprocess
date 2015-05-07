.. -*- mode: rst -*-

.. image:: https://travis-ci.org/neurospin/pypreprocess.svg?branch=master
   :target: https://travis-ci.org/neurospin/pypreprocess
   :alt: Build Status
   
.. image:: https://coveralls.io/repos/dohmatob/pypreprocess/badge.svg?branch=master
   :target: https://coveralls.io/r/dohmatob/pypreprocess?branch=master
   
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
First install the above dependencies by copy-pasting the following commands in a terminal:

      wget -O- http://neuro.debian.net/lists/precise.us-nh.libre | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
      && sudo apt-key adv --recv-keys --keyserver pgp.mit.edu 2649A5A9
      
      sudo apt-get update
      && sudo apt-get install python-scipy python-nose python-nibabel python-sklearn python-sympy python-networkx python-nipy python-nipype
      && sudo apt-get install python-pip && pip install nilearn --user


Then install pypreprocess itself by running:

       python setup.py install --user


Use-case Examples
=================
We have written some examplary scripts for preprocessing some popular datasets.
The **examples** directory contains a set of scripts, each demoing an aspect of pypreprocessing. Some scripts even provide use-cases for the nipy-based GLM. The examples use publicly available sMRI and fMRI data. Data fetchers are based on the nilearn API.
The main examples scripts can be summarized as follows:

* **examples/nipype_preproc_spm_auditory.py**: demos preprocessing + first-level GLM (using nipy)  on the single-subject SPM auditory dataset.

* **examples/nipype_preproc_spm_multimodal_faces.py**: demos preprocessing + first-level fixed-effects GLM on R. Henson's multi-modal face dataset (multiple sessions)

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


Using .ini configuration files to specify pipeline
==================================================
It's possible (and recommended) to configure the preprocessing pipeline just by copying
any of the `.ini` configuration files under the `examples` sub-directory and modifying it accordingly (usually, you only need to modify the `dataset_dir` paramter),
and then run

      `python pypreprocess.py your.ini`
      
Alternatively, you can do this

      `preproc_output = do_subjects_preproc("your.ini", dataset_dir=dataset_dir)`

where the optional ``dataset_dir=dataset_dir`` parameter (you can specify more optional parameters, BTW) simply overrides any value specified in the .ini file.

Pipelines
=========
We have put in place two main pipelines for preprocessing: the *standard* pipeline, and the *DARTEL*-based pipeline. In the end of either method, each subject's EPI data has been corrected for artefacts, and placed into the same reference space (MNI).
When you invoke the ``do_subjects_preproc(..)`` API of [nipype_preproc_spm_utils.py](https://github.com/neurospin/pypreprocess/blob/master/nipype_preproc_spm_utils.py) to preprocess a dataset (group of subjects), the default pipeline used is the standard one; passing the option ``do_dartel=True`` forces the DARTEL-based pipeline to be used.
Also you can fine-tune your pipeline using the the various supported parameters in you .ini file (see the ``examples/`` subdirectory for examples).

Standard pipeline
-----------------
For each subject, the following preprocessing steps are undergone:

* Motion correction is done to estimate and correct for subject's head motion during the acquisition.

* The subject's anatomical image is coregistered against their fMRI images (precisely, to the mean thereof). Coregistration is important as it allows deformations of the anatomy to be directly applicable to the fMRI, or for ROIs to be defined on the anatomy.

* Tissue Segmentation is then employed to segment the anatomical image into GM, WM, and CSF compartments by using TPMs (Tissue Probability Maps) as priors.

* The segmented anatomical image are then warped into the MNI template space by applying the deformations learned during segmentation. The same deformations have been applied to the fMRI images.

DARTEL pipeline
---------------
Motion correction, and coregistration go on as for the standard pipeline. The only difference is the way the subject EPI are warped into MNI space. viz:
* Group/Inter-subject Normalization is done using the SPM8 [DARTEL](http://www.fil.ion.ucl.ac.uk/spm/software/spm8/SPM8_Release_Notes.pdf) to warp subject brains into MNI space. The idea is to register images by computing a “flow field” which can then be “exponentiated” to generate both forward and backward deformations. Processing begins with the “import” step. This involves taking the parameter files produced by the segmentation (NewSegment), and writing out rigidly transformed versions of the tissue class images, such that they are in as close alignment as possible with the tissue probability maps.   The next step is the registration itself. This involves the simultaneous registration of e.g. GM with GM, WM with WM and 1-(GM+WM) with 1-(GM+WM) (when needed, the 1- (GM+WM) class is generated implicitly, so there is no need to include this class yourself). This procedure begins by creating a mean of all the images, which is used as an initial template. Deformations from this template to each of the individual images are computed, and the template is then re-generated by applying the inverses of the deformations to the images and averaging. This procedure is repeated a number of times.  Finally, warped versions of the images (or other images that are in alignment with them) can be generated.
[nipype_preproc_spm_abide.py](https://github.com/neurospin/pypreprocess/blob/master/abide/nipype_preproc_spm_abide.py) is a script which uses this pipeline to preprocess the [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/).

Intra-subject preprocessing in pure Python (with no compiled code, etc.)
========================================================================
A couple of modules for intra-subject preprocessing (slice-timing correction, motion-correction, coregistration, etc.)
in pure (only using builtins and numpy/scipy official stuff, no compiled code, no wrappers) Python have been implemented.
To demo this feature, cd to the **pypreprocess/examples** directory, and run the following command:

       python purepython_preproc_demo.py

Development
===========
You can check the latest version of the code with the command::

       git clone git://github.com/neurospin/pypreprocess.git

or if you have write privileges::

       git clone git@github.com:neurospin/pypreprocess.git
