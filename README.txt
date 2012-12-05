
What is this ?
==============
pypreprocess is a collection of python scripts for preprocessing (motion 
correction, spatial normalization, smoothing, etc.) fMRI data using 
nipype's SPM and FSL interfaces. It also contains utilities for automatic 
QA like registration checks (using nipy.labs), and html report generation 
(using tempita and jquery).

Check out the wiki at https://github.com/neurospin/pypreprocess/wiki.


REQUIREMENTS
============
First of all, you will need to have the following (standard) tools 
installed on your system:
* git
* mercurial (hg)
* pip

The requirements/dependencies (nipy, nipype, traits, nibabel, nisl, etc.) 
are documented in the *dependencies.txt files.

To install these dependencies in one go, simply chdir to the directory 
containing this README.txt and then type (in your terminal):

	python install_depenencies.py


TODO
====
Improve on this README.

-
DED
