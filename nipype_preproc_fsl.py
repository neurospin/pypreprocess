"""
Author: Bertrand Thirion, Alexandre Abraham
"""
# Standard imports
import os
import shutil

# Nipype imports
from nipype.caching import Memory
import nipype.interfaces.fsl as fsl

from nisl import datasets

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
n_subjects = 3 

# create directories where to write the image
# for directory in ['/tmp/nyu/', '/tmp/nyu/fmri/', '/tmp/nyu/t1/']:
#     if not os.path.exists(directory):
#         os.mkdir(directory)

# cmd_prefix = 'fsl5.0-'
# nyu = datasets.fetch_nyu_rest(n_subjects=n_subjects)

##############################################################
# Preprocess functional data
##############################################################


def do_subject_preproc(subject_output_dir,
                         anat_image,
                         fmri_images,
                         cmd_prefix=None,
                         **kwargs):

    # prepare for smart-caching
    t1_dir = os.path.join(subject_output_dir, 't1')
    if not os.path.exists(t1_dir):
        os.makedirs(t1_dir)
    cache_dir = os.path.join(subject_output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    #  Brain Extraction
    bet = fsl.BET(
            in_file=fmri_images,
            out_file=os.path.join(subject_output_dir, 'fmri_bet.nii.gz'))
    if cmd_prefix:
        bet._cmd = cmd_prefix + bet._cmd
    bet.run()

    #  Slice timing correction
    st = fsl.SliceTimer(
            in_file=os.path.join(subject_output_dir, 'fmri_bet.nii.gz'),
            out_file=os.path.join(subject_output_dir, 'fmri_st.nii.gz'))
    st.inputs.interleaved = True
    if cmd_prefix:
        st._cmd = cmd_prefix + st._cmd
    result_st = st.run()

    #  Motion correction
    mcflt = fsl.MCFLIRT(
        in_file=os.path.join(subject_output_dir, 'fmri_st.nii.gz'),
        cost='mutualinfo',
        out_file = os.path.join(subject_output_dir, 'fmri_mc.nii.gz'))
    if cmd_prefix:
        mcflt._cmd = cmd_prefix + mcflt._cmd
    mcflt.run()

    # Coregistration
    flt1 = fsl.FLIRT(in_file=os.path.join(subject_output_dir, 'fmri_mc.nii.gz'), 
            reference=anat_image,
            out_file=os.path.join(subject_output_dir, 'fmri_coreg_anat.nii.gz'),
            out_matrix_file=os.path.join(subject_output_dir, 'fmri_flirt.mat'))
    if cmd_prefix:
        flt1._cmd = cmd_prefix + flt1._cmd
    flt1.run()


    #  T1 Spatial normalization
    flt2 = fsl.FLIRT(in_file=anat_image, 
            reference="/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz",
            out_file=os.path.join(t1_dir, 't1_warped.nii.gz'),
            out_matrix_file=os.path.join(t1_dir, 't1_flirt.mat'))
    if cmd_prefix:
        flt2._cmd = cmd_prefix + flt2._cmd
    flt2.run()

    #  Concatenate transformations
    concat = fsl.ConvertXFM(
            in_file=os.path.join(subject_output_dir, 'fmri_flirt.mat'),
            in_file2=os.path.join(t1_dir, 't1_flirt.mat'),
            concat_xfm=True,
            out_file=os.path.join(subject_output_dir, 'fmri_coreg.mat'))
    if cmd_prefix:
        concat._cmd = cmd_prefix + concat._cmd
    concat.run()

    #  Functional/T1 coregistration
    flt = fsl.ApplyXfm(
            in_file=os.path.join(subject_output_dir, 'fmri_mc.nii.gz'),
            in_matrix_file=os.path.join(subject_output_dir, 'fmri_coreg.mat'),
            reference="/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz",
            out_file=os.path.join(subject_output_dir, 'fmri_coreg.nii.gz'),
            out_matrix_file=os.path.join(subject_output_dir, 'fmri_final.mat'))
    if cmd_prefix:
        flt._cmd = cmd_prefix + flt._cmd
    flt.run()


# mainpath = '/tmp/nyu'
# if not os.path.exists(mainpath):
#     os.mkdir(mainpath)
# for i, func in enumerate(nyu.func):
#     path = os.path.join(mainpath, str(i))
#     if not os.path.exists(path):
#         os.mkdir(path)
#     do_subject_preproc(path, nyu.anat_anon[i], func, cmd_prefix='fsl5.0-')

# """
# fnt = fsl.FNIRT(in_file='/tmp/kr080082/t1/kr080082_t1.nii.gz', 
#                 ref_file='/tmp/T1.nii', #ref_file=example_data('mni.nii'),
#                 warped_file='/tmp/kr080082/t1/kr080082_t1_warped.nii.gz')
# if neurospin:
#     fnt._cmd = 'fsl4.1-' + fnt._cmd
# res = fnt.run()
# """
