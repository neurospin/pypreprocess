import os
import nibabel
from nose.tools import assert_equal, assert_true, assert_false
import nose
from ._test_utils import create_random_image, _make_sd, _save_img

DATA_DIR = "test_tmp_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


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


def test_not_unique_func_filenames_exception_thrown():
    sd = _make_sd(func_filenames=["/tmp/titi/func1.nii",
                                  "/tmp/titi/func2.nii"],
                  output_dir="/tmp")
    try:
        sd.sanitize()
        raise RuntimeError("Check failed!")
    except RuntimeError:
        pass

    sd = _make_sd(func_filenames=["/tmp/titi/session1/func.nii",
                           "/tmp/titi/session1/func.nii"],
                     output_dir="/tmp")
    try:
        sd.sanitize()
        raise RuntimeError("Check failed!")
    except RuntimeError:
        pass

    sd = _make_sd(func_filenames=[["/tmp/titi/func/1.img",
                                   "/tmp/titi/func/2.img"],
                           ["/tmp/titi/func/1.img", "/tmp/titi/func/3.img"]],
                     output_dir="/tmp")
    try:
        sd.sanitize()
        raise RuntimeError("Check failed!")
    except RuntimeError:
        pass

    sd = _make_sd(func_filenames=["/tmp/titi/func/1.img",
                           ["/tmp/titi/func/1.img", "/tmp/titi/func/3.img"]],
                     output_dir="/tmp")
    try:
        sd.sanitize()
        raise RuntimeError("Check failed!")
    except RuntimeError:
        pass

    sd = _make_sd(func_filenames=[["/tmp/titi/func/1.img",
                                   "/tmp/titi/func/2.img"],
                           ["/tmp/titi/func/3.img", "/tmp/titi/func/4.img"]],
                     output_dir="/tmp")
    sd.sanitize()

    # abspaths of func images should be different with a session
    sd = _make_sd(func_filenames=[["/tmp/titi/func/1.img",
                            "/tmp/titi/func/1.img"],
                           ["/tmp/titi/func/3.img",
                            "/tmp/titi/func/4.img"]],
                     output_dir="/tmp")
    try:
        sd.sanitize()
        raise RuntimeError("Check failed!")
    except RuntimeError:
        pass

def test_issue_40():
    sd = _make_sd(func_filenames=[['/tmp/rob/ds005/pypreprocess_output/sub001/task001_run001/deleteorient_1_bold.nii', '/tmp/rob/ds005/pypreprocess_output/sub001/task001_run001/deleteorient_1_bold.nii', '/tmp/rob/ds005/pypreprocess_output/sub001/task001_run001/deleteorient_1_bold.nii']],
                     output_dir="/tmp")
    try:
        sd.sanitize()
        raise RuntimeError("Check failed!")
    except RuntimeError:
        pass

# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
