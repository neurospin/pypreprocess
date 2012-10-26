"""
:Module: nipype_preproc_spm_utils
:Synopsis: routine functions for SPM preprocessing business
:Author: dohmatob elvis dopgima

BUGFIX: Somewhere in the SPM8 Matlab backend, nifti input images are 
expected in the format "somenifti.nii,1" instead of "somenifti.nii" 
(notice the "1" in the former); otherwise --and for some wierd 
reason-- the backend is dreadfully slow (about 20*60 times!). As a 
quick (and dirty, of course!) fix, I have created patched versions 
_Realign, etc., of the spm.Realign, etc., classes by overriding the
SPMCommand._make_matlab_command method with a patched version. I'll 
investigate this bug (matlab or nipype ?) latter.
"""

# standard imports
import os
import shutil
import re

# import useful interfaces from nipype
from nipype.caching import Memory
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab

# set matlab exec path
MATLAB_EXEC = "/neurospin/local/matlab/bin/matlab"
if 'MATLAB_EXEC' in os.environ:
    MATLAB_EXEC = os.environ['MATLAB_EXEC']
assert os.path.exists(MATLAB_EXEC), \
    "nipype_preproc_smp_utils: MATLAB_EXEC: %s, \
doesn't exist; you need to export MATLAB_EXEC" % MATLAB_EXEC
matlab.MatlabCommand.set_default_matlab_cmd(MATLAB_EXEC)

# set matlab SPM back-end path
MATLAB_SPM_DIR = '/i2bm/local/spm8'
if 'MATLAB_SPM_DIR' in os.environ:
    MATLAB_SPM_DIR = os.environ['MATLAB_SPM_DIR']
assert os.path.exists(MATLAB_SPM_DIR), \
    "nipype_preproc_smp_utils: MATLAB_SPM_DIR: %s,\
 doesn't exist; you need to export MATLAB_SPM_DIR" % MATLAB_SPM_DIR
matlab.MatlabCommand.set_default_paths(MATLAB_SPM_DIR)

# set template
T1_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'templates/T1.nii')
if 'T1_TEMPLATE' in os.environ:
    T1_TEMPLATE = os.environ['T1_TEMPLATE']
assert os.path.exists(MATLAB_SPM_DIR), \
    "nipype_preproc_smp_utils: T1_TEMPLATE: %s, doesn't exist; \
you need to export T1_TEMPLATE" % T1_TEMPLATE

class _Realign(spm.Realign):
    def _make_matlab_command(self, contents, postscript=None):
        mscript = spm.Realign._make_matlab_command(self, contents, postscript=postscript)

        patched_mscript = re.sub("\.nii", ".nii,1", mscript)

        return patched_mscript

class _Coregister(spm.Coregister):
    def _make_matlab_command(self, contents, postscript=None):
        mscript = spm.Coregister._make_matlab_command(self, contents, postscript=postscript)
       
        patched_mscript = re.sub("\.nii", ".nii,1", mscript)

        return patched_mscript

class _Segment(spm.Segment):
    def _make_matlab_command(self, contents, postscript=None):
        mscript = spm.Segment._make_matlab_command(self, contents, postscript=postscript)

        patched_mscript = re.sub("\.nii", ".nii,1", mscript)

        return patched_mscript

class _Normalize(spm.Normalize):
    def _make_matlab_command(self, contents, postscript=None):
        mscript = spm.Normalize._make_matlab_command(self, contents, postscript=postscript)

        patched_mscript = re.sub("\.nii", ".nii,1", mscript)

        return patched_mscript

def do_subject_preproc(subject_output_dir,
                       anat_image,
                       fmri_images,
                       check_preproc=True,
                       **kwargs):
    """
    Function preprocessing data for a single subject.

    """
    # prepare for smart-caching
    t1_dir = os.path.join(subject_output_dir, 't1')
    if not os.path.exists(t1_dir):
        os.makedirs(t1_dir)
    cache_dir = os.path.join(subject_output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    #  motion correction
    realign = mem.cache(_Realign)
    realign_result = realign(in_files=fmri_images,
                             register_to_mean=True,
                             mfile=True,
                             jobtype='estwrite')

    if check_preproc:
        # verify that motion correction did well
        pass

    # co-registration against structural (anatomical)
    coreg = mem.cache(_Coregister)
    coreg_result = coreg(target=realign_result.outputs.mean_image,
                         source=anat_image,
                         apply_to_files=realign_result.outputs.realigned_files,
                         jobtype='estwrite')

    if check_preproc:
        # verify that coregistration did well
        pass

    # # learn the deformation on T1 MRI without segmentation
    # normalize = mem.cache(_Normalize)
    # norm_result = normalize(source=coreg_result.outputs.coregistered_source,
    #                         template=T1_TEMPLATE)

    #  alternative: Segmentation & normalization
    normalize = mem.cache(_Normalize)
    segment = mem.cache(_Segment)
    segment_result = segment(data=coreg_result.outputs.coregistered_source,
                             gm_output_type=[True, True, True],
                             wm_output_type=[True, True, True],
                             csf_output_type=[True, True, True])

    # deform FRMI images unto T1 template
    norm_apply = normalize(
        parameter_file=segment_result.transformation_mat,
        apply_to_files=realign_result.outputs.realigned_files,
        jobtype='write',
        write_voxel_sizes=[3, 3, 3])

    # wfmri = norm_apply.outputs.normalized_files
    # if type(wfmri) is str:
    #     wfmri = [wfmri]
    # for wf in wfmri:
    #     shutil.copyfile(wf, os.path.join(subject_output_dir,
    #                                      os.path.basename(wf)))

    # deform anat image unto T1 template
    norm_apply = normalize(
        parameter_file=segment_result.transformation_mat,
        apply_to_files=coreg_result.outputs.coregistered_source,
        jobtype='write',
        write_voxel_sizes=[1, 1, 1])

    # wanat = norm_apply.outputs.normalized_files
    # shutil.copyfile(wanat, os.path.join(t1_dir, os.path.basename(wanat)))
