import nose
import nose.tools
import os
import nibabel
from ._test_utils import create_random_image
from ..nipype_preproc_spm_utils_bis import (niigz2nii,
                                            SubjectData,
                                            _do_subject_realign
                                            )
from ..datasets import fetch_spm_auditory_data,


def test_subject_data():
    # create subject data
    sd = SubjectData()
    sd.subject_id = 'sub001'
    sd.output_dir = os.path.join("/tmp/kimbo/", sd.subject_id)
    sd.func = '/tmp/func.nii.gz'
    nibabel.save(create_random_image(ndim=4), sd.func)
    sd.anat = '/tmp/anat.nii.gz'
    nibabel.save(create_random_image(), sd.anat)

    # sanitize subject data
    sd.sanitize()

    # checks
    nose.tools.assert_equal(sd.func, ['/tmp/kimbo/sub001/func.nii'])
    nose.tools.assert_equal(sd.anat, '/tmp/kimbo/sub001/anat.nii')
    nose.tools.assert_equal(sd.session_id, ['session_0'])


def test_do_subject_realign():
    sd = fetch_spm_auditory_data(os.path.join(os.environ['HOME'],
                                              "CODE/datasets/spm_auditory"))
    subject_data = SubjectData(func=sd.func, anat=sd.anat,
                               output_dir="/tmp/toto/sub001")
    subject_data = _do_subject_realign(subject_data)

# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
