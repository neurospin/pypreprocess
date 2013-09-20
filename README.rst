	
What is this ?
==============
**pypreprocess** is a collection of python scripts for preprocessing (motion 
correction, spatial normalization, smoothing, etc.) fMRI data using 
nipype's SPM and FSL interfaces. It also contains utilities for automatic 
QA like registration checks (using nipy.labs), and template-based html report
generation using (tempita, jquery, and home-grown css).

Check out the wiki at https://github.com/neurospin/pypreprocess/wiki.


REQUIREMENTS
============
First of all, you will need to have the following installed:
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
	

The requirements/dependencies (nipy, nipype, traits, nibabel, etc.) 
are documented in the *dependencies.txt files.

To install these dependencies in one go, simply chdir to the directory 
containing this README and then type (in your terminal):

	python install_depenencies.py


Use-case Examples
=================
We have written some examplary scripts for preprocessing some popular datasets.


SPM auditory (single-subject)
-----------------------------
cd to the pypreprocess/examples directory, and run the following command:

       python nipy_glm_spm_auditory.py spm_auditory spm_auditory_runs/ 

Now open the file spm_auditory_runs/subXYZ/report.html in your browser (firefox), to see
the generate report (QA).

'Serious' examples
----------------
The scripts/ sub-folder contains scripts for preprocessing popular datasets like ABIDE, HCP, HAXBY2001, NYU rest, etc.
They should work 'out-of-the-box'.

TODO
====
Improve on this README.

Enjoy!
