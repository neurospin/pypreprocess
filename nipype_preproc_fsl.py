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
n_subjects = 12

# create directories where to write the image
for directory in ['/tmp/nyu/', '/tmp/nyu/fmri/', '/tmp/nyu/t1/']:
    if not os.path.exists(directory):
        os.mkdir(directory)

cmd_prefix = 'fsl5.0-'
nyu = datasets.fetch_nyu_rest(n_subjects=n_subjects)

# Reference image
ref = nyu.anat_skull[0]

##############################################################
# Preprocess functional data
##############################################################


def do_subject_preproc(subject_output_dir,
                         anat_image,
                         fmri_images,
                         **kwargs):

    # prepare for smart-caching
    t1_dir = os.path.join(subject_output_dir, 't1')
    if not os.path.exists(t1_dir):
        os.makedirs(t1_dir)
    cache_dir = os.path.join(subject_output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    #  Slice timing correction
    st = fsl.SliceTimer()
    st.inputs.in_file = fmri_images
    st.inputs.interleaved = True
    st._cmd = cmd_prefix + st._cmd
    result_st = mem.cache(st.run(), )
    shutil.move(str(result_st.outputs).split(' ')[-1].split('\n')[0],
            os.path.join(subject_output_dir, 'fmri_st.nii.gz'))

    #  Motion correction
    mcflt = fsl.MCFLIRT(
        in_file=os.path.join(subject_output_dir, 'fmri_st.nii.gz'),
        cost='mutualinfo')
    mcflt.out_file = os.path.join(subject_output_dir, 'fmri_st_mc.nii.gz')
    mcflt._cmd = cmd_prefix + mcflt._cmd
    mcflt.run()

    #  T1 Spatial normalization
    flt = fsl.FLIRT(in_file=nyu.anat_image, 
                    reference=ref,
                    out_file=os.path.join(subject_output_dir, 'fmri_st_mc_sn.nii.gz'),
                    out_matrix_file=os.path.join(subject_output_dir, 'fmri_sn.mat'))
    flt._cmd = cmd_prefix + flt._cmd
    flt.run()

    #  Functional/T1 coregistration
    flt = fsl.FLIRT(
        in_file=os.path.join(subject_output_dir, 'fmri_st_mc_sn.nii.gz'),
        reference="/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz",
        out_file=os.path.join(subject_output_dir, 'fmri_st_mc_sn_coreg.nii.gz'),
        out_matrix_file=os.path.join(subject_output_dir, 'fmri_coreg.mat'))
    flt._cmd = cmd_prefix + flt._cmd
    flt.run()


for i, func in enumerate(nyu.func):
    path = '/tmp/nyu/' + str(i)
    os.mkdir('/tmp/nyu/' + str(i))
    do_subject_preproc(path, func, nyu.anat_anon[i])

"""
    #  Apply to fMRI
    applyxfm = fsl.ApplyXfm()
    applyxfm.inputs.in_file = \
        "/tmp/nyu/fmri/nyu_subject%d_st_mcf.nii.gz" % index
    applyxfm.inputs.in_matrix_file = '/tmp/kr080082/t1/kr080082_t1_flirt.mat'
    applyxfm.inputs.out_file = \
        '/tmp/kr080082/fmri/kr080082_series00000_st_mcf_warped.nii.gz'
    applyxfm.inputs.reference = '/tmp/T1.nii'
    applyxfm.inputs.apply_xfm = True
    if neurospin:
        applyxfm._cmd = 'fsl4.1-' + applyxfm._cmd
    result = applyxfm.run() 
"""
"""
fnt = fsl.FNIRT(in_file='/tmp/kr080082/t1/kr080082_t1.nii.gz', 
                ref_file='/tmp/T1.nii', #ref_file=example_data('mni.nii'),
                warped_file='/tmp/kr080082/t1/kr080082_t1_warped.nii.gz')
if neurospin:
    fnt._cmd = 'fsl4.1-' + fnt._cmd
res = fnt.run()
"""
# Fixme: get the template from nipype ? 
