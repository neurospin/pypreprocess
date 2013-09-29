import os
import glob
import itertools
from pypreprocess.nipype_preproc_spm_utils_bis import (do_subjects_preproc,
                                                       SubjectData
                                                       )
from pypreprocess.datasets import (fetch_spm_auditory_data,
                                   fetch_spm_multimodal_fmri_data
                                   )


def _abide_factory(institute="KKI"):
    for scans in sorted(glob.glob(
            "/home/elvis/CODE/datasets/ABIDE/%s_*/%s_*/scans" % (
                institute, institute))):
        subject_data = SubjectData()
        subject_data.subject_id = os.path.basename(os.path.dirname(
                os.path.dirname(scans)))
        subject_data.func = os.path.join(scans,
                                         "rest/resources/NIfTI/files/rest.nii")
        subject_data.anat = os.path.join(
            scans, "anat/resources/NIfTI/files/mprage.nii")
        subject_data.output_dir = os.path.join(os.environ['HOME'],
                                               "CODE/datasets/abide_preproc",
                                               subject_data.subject_id)
        yield subject_data

# run preproc pipeline
do_subjects_preproc(_abide_factory(), fwhm=[8, 8, 8])

if 0x1:
    for (with_anat, do_segment, do_normalize,
         fwhm, hard_link_output) in itertools.product(
        [False, True], [False, True], [False, True], [0, 8, [8, 8, 8]],
        [False, True]):
        # load spm auditory data

        sd = fetch_spm_auditory_data(os.path.join(
                os.environ['HOME'], 'CODE/datasets/spm_auditory'))
        subject_data1 = SubjectData(func=[sd.func],
                                    anat=sd.anat if with_anat else None)
        subject_data1.output_dir = "/tmp/kimbo/sub001/"

        # load spm multimodal fmri data
        sd = fetch_spm_multimodal_fmri_data(os.path.join(
                os.environ['HOME'], 'CODE/datasets/spm_multimodal_fmri'))
        subject_data2 = SubjectData(func=[sd.func1, sd.func2],
                                    anat=sd.anat if with_anat else None,
                                   session_id=['Session 1', "Session 2"])
        subject_data2.output_dir = "/tmp/kiki/sub001/"

        do_subjects_preproc([subject_data1, subject_data2],
                            do_segment=do_segment,
                            do_normalize=do_normalize,
                            hard_link_output=hard_link_output,
                            fwhm=fwhm
                            )
