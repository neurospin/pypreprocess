from pypreprocess.subject_data import SubjectData
from nose.tools import assert_equal, assert_true, assert_false
import nibabel
from ._test_utils import create_random_image
import os

DATA_DIR = "test_tmp_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def test_sujectdata_init():
    sd = SubjectData(anat='/tmp/anat.nii.gz', func='/tmp/func.nii.gz')
    assert_equal(sd.anat, "/tmp/anat.nii.gz")
    assert_equal(sd.func, "/tmp/func.nii.gz")


def test_sujectdata_sanitize():
    func = create_random_image(ndim=4)
    anat = create_random_image(ndim=3)
    func_filename = '%s/func.nii.gz' % DATA_DIR
    anat_filename = '%s/anat.nii.gz' % DATA_DIR
    nibabel.save(func, func_filename)
    nibabel.save(anat, anat_filename)
    sd = SubjectData(anat=anat_filename,
                     func=func_filename,
                     output_dir="/tmp/titi")
    sd.sanitize()
