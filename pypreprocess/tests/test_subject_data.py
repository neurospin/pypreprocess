import os
import nibabel
from pypreprocess.subject_data import SubjectData
from nose.tools import assert_equal, assert_true, assert_false
import nose
from ._test_utils import create_random_image

DATA_DIR = "test_tmp_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def _make_sd(ext=".nii.gz", n_sessions=1, make_sess_dirs=False,
             unique_func_names=False):
    func = [create_random_image(ndim=4) for _ in xrange(n_sessions)]
    anat = create_random_image(ndim=3)
    anat_filename = '%s/anat%s' % (DATA_DIR, ext)
    nibabel.save(anat, anat_filename)
    func_filenames = []
    for sess in xrange(n_sessions):
        sess_dir = DATA_DIR if not make_sess_dirs else os.path.join(
            DATA_DIR, "session%i" % sess)
        if not os.path.exists(sess_dir):
            os.makedirs(sess_dir)
        func_filename = '%s/func%s%s' % (
            sess_dir, "_sess_%i_" % sess if (
                n_sessions > 1 and unique_func_names) else "", ext)
        nibabel.save(func[sess], func_filename)
        func_filenames.append(func_filename)
    sd = SubjectData(anat=anat_filename,
                     func=func_filenames,
                     output_dir="/tmp/titi")

    return sd


def test_sujectdata_init():
    sd = SubjectData(anat='/tmp/anat.nii.gz', func='/tmp/func.nii.gz')
    assert_equal(sd.anat, "/tmp/anat.nii.gz")
    assert_equal(sd.func, "/tmp/func.nii.gz")


def test_sujectdata_sanitize():

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


def test_unique_func_filenames():
    # XXX against issue 40
    for ext in [".nii", ".nii.gz"]:
        for make_sess_dirs in [False, True]:
            for n_sessions in [1, 2]:
                for niigz2nii in [False, True]:
                    sd = _make_sd(ext=ext, n_sessions=n_sessions,
                                  make_sess_dirs=make_sess_dirs,
                                  unique_func_names=not make_sess_dirs)
                    sd.sanitize(niigz2nii=niigz2nii)

                    assert_equal(len(sd.func), len(set(sd.func)))

    return sd

# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
