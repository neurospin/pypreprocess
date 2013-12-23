import os
import nibabel
from pypreprocess.subject_data import SubjectData
from nose.tools import assert_equal, assert_true, assert_false
import nose
from ._test_utils import create_random_image

DATA_DIR = "test_tmp_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def test_sujectdata_init():
    sd = SubjectData(anat='/tmp/anat.nii.gz', func='/tmp/func.nii.gz')
    assert_equal(sd.anat, "/tmp/anat.nii.gz")
    assert_equal(sd.func, "/tmp/func.nii.gz")


def test_sujectdata_sanitize():
    def _make_sd(ext=".nii.gz"):
        func = create_random_image(ndim=4)
        anat = create_random_image(ndim=3)
        func_filename = '%s/func%s' % (DATA_DIR, ext)
        anat_filename = '%s/anat%s' % (DATA_DIR, ext)
        nibabel.save(func, func_filename)
        nibabel.save(anat, anat_filename)
        sd = SubjectData(anat=anat_filename,
                         func=func_filename,
                         output_dir="/tmp/titi")

        return sd

    sd = _make_sd(ext=".nii.gz")
    sd.sanitize()
    assert_equal(os.path.basename(sd.func[0]), "func.nii.gz")
    assert_equal(os.path.basename(sd.anat), "anat.nii.gz")

    sd = _make_sd(ext=".nii.gz")
    sd.sanitize(niigz2nii=True)
    assert_equal(os.path.basename(sd.func[0]), "func.nii")
    assert_equal(os.path.basename(sd.anat), "anat.nii")

    sd = _make_sd(ext=".nii")
    sd.sanitize()
    assert_equal(os.path.basename(sd.func[0]), "func.nii")
    assert_equal(os.path.basename(sd.anat), "anat.nii")

    sd = _make_sd(ext=".nii")
    sd.sanitize(niigz2nii=True)
    assert_equal(os.path.basename(sd.func[0]), "func.nii")
    assert_equal(os.path.basename(sd.anat), "anat.nii")

# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
