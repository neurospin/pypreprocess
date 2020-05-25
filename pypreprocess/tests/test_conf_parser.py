import os
import pytest
from ..conf_parser import _generate_preproc_pipeline
from ._test_utils import _make_sd


def _make_config(out_file, **kwargs):
    config = """
[config]
"""
    for k, v in kwargs.items():
        if isinstance(v, list):
            v = ", ".join(v)
        config += "%s=%s\r\n" % (k, v)
    fd = open(out_file, "w")
    fd.write(config)
    return config


def test_obligatory_params_config():
    dataset_dir = "/tmp/data"
    output_dir = "/tmp/output"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    config_file = os.path.join(dataset_dir, "empty.ini")
    _make_config(config_file)
    with pytest.raises(ValueError):
        assert "dataset_dir not specified" in _generate_preproc_pipeline(config_file)

    _make_config(config_file, dataset_dir=dataset_dir)
    with pytest.raises(ValueError):
        assert "output_dir not specified" in _generate_preproc_pipeline(config_file)

    # this should not give any errors
    _make_config(config_file, dataset_dir=dataset_dir, output_dir=output_dir,
                 session_1_func="fM00223/fM00223_*.img")
    _generate_preproc_pipeline(config_file)


def test_issue110():
    dataset_dir = "/tmp/data"
    output_dir = "/tmp/output"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    config_file = os.path.join(dataset_dir, "empty.ini")
    _make_config(config_file, newsegment=True, dataset_dir=dataset_dir,
                 output_dir=output_dir)
    _, options = _generate_preproc_pipeline(config_file)
    assert options["newsegment"]


def test_empty_params_default_to_none():
    output_dir = "/tmp/output"
    config_file = "/tmp/empty.ini"
    for i in range(3):
        dataset_dir = " " * i
        _make_config(config_file, dataset_dir=dataset_dir,
                     output_dir=output_dir)
        _make_config(config_file)
        with pytest.raises(ValueError):
            assert "dataset_dir not specified" in _generate_preproc_pipeline(config_file)

def test_bf_issue_62():
    dataset_dir = "/tmp/dataset"
    output_dir = "/tmp/output"
    config_file = os.path.join(dataset_dir, "conf.ini")
    _make_sd(func_filenames=[os.path.join(dataset_dir,
                                          "sub001/session1/func.nii"),
                             os.path.join(dataset_dir,
                                          "sub001/session2/func.nii"),
                             os.path.join(dataset_dir,
                                          "sub001/session3/func.nii")],
             output_dir=os.path.join(output_dir, "sub001"))
    _make_sd(func_filenames=[os.path.join(dataset_dir,
                                          "sub002/session1/func.nii"),
                             os.path.join(dataset_dir,
                                          "sub002/session2/func.nii")],
             output_dir=os.path.join(output_dir, "sub002"))
    _make_config(config_file, dataset_dir=dataset_dir, output_dir=output_dir,
                 session_1_func="session1/func.nii",
                 session_2_func="session2/func.nii",
                 session_3_func="session3/func.nii")
    subjects, _ = _generate_preproc_pipeline(config_file)
    assert len(subjects[0]['func']) == 3
    assert len(subjects[1]['func']) == 2


def test_newsegment_if_dartel():
    dataset_dir = "/tmp/dataset"
    output_dir = "/tmp/output"
    config_file = os.path.join(dataset_dir, "conf.ini")
    for kwargs in [{}, dict(newsegment=True), dict(newsegment=False)]:
        _make_config(config_file, dataset_dir=dataset_dir,
                     output_dir=output_dir, dartel=True, **kwargs)
        _, params = _generate_preproc_pipeline(config_file)
        assert params["dartel"]
        assert params["newsegment"]


def test_bf_issue_122():
    dataset_dir = "/tmp/dataset"
    output_dir = "/tmp/output"
    config_file = os.path.join(dataset_dir, "conf.ini")
    _make_config(config_file, dataset_dir=dataset_dir, output_dir=output_dir,
                 session_1_func="session1/func_3D.nii")
    _make_sd(func_filenames=[os.path.join(dataset_dir,
                                          "sub001/session1/func_3D.nii")],
             output_dir=os.path.join(output_dir, "sub001"),
             func_ndim=3)
    subjects, _ = _generate_preproc_pipeline(config_file)
    subjects[0].sanitize()  # 122 reports a bug here


def test_env_vars():
    os.environ["DATASET_DIR"] = "/tmp/dataset/"
    os.environ["OUTPUT_DIR"] = "/tmp/output/"
    config_file = os.path.join(os.environ["DATASET_DIR"], "conf.ini")
    _make_config(config_file, session_1_func="session1/func.nii")
    _generate_preproc_pipeline(config_file)  # this call shouldn't crash


def test_explicit_list_subdirs():
    dataset_dir = "/tmp/dataset"
    output_dir = "/tmp/output"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    config_file = os.path.join(dataset_dir, "empty.ini")
    _make_config(config_file, subject_dirs=["sub01", "sub02"],
                 session_1_func="session1/func_3D.nii")
    for subject_id in ["sub01", "sub02"]:
        _make_sd(func_filenames=[os.path.join(
            dataset_dir, "%s/session1/func_3D.nii" % subject_id)],
                 output_dir=os.path.join(output_dir, subject_id))
    subjects, _ = _generate_preproc_pipeline(config_file)
    assert len(subjects) == 2


def test_list_of_subj_wildcards():
    dataset_dir = "/tmp/dataset"
    output_dir = "/tmp/output"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    for subject_id in ["subx01", "subx02", "suby01", "suby02"]:
        _make_sd(func_filenames=[os.path.join(
            dataset_dir, "%s/session1/func_3D.nii" % subject_id)],
                 output_dir=os.path.join(output_dir, subject_id))
    config_file = os.path.join(dataset_dir, "empty.ini")
    _make_config(config_file, subject_dirs=["subx*", "suby*"],
                 session_1_func="session1/func_3D.nii")
    subjects, _ = _generate_preproc_pipeline(config_file)
    assert len(subjects) == 4


def test_user_specified_scratch():
    dataset_dir = "/tmp/dataset"
    output_dir = "/tmp/output"
    scratch = "/tmp/scratch"
    config_file = os.path.join(dataset_dir, "conf.ini")
    _make_config(config_file, dataset_dir=dataset_dir, output_dir=output_dir,
                 scratch=scratch, session_1_func="session1/func_3D.nii")
    _make_sd(func_filenames=[os.path.join(dataset_dir,
                                          "sub001/session1/func_3D.nii")],
             output_dir=os.path.join(output_dir, "sub001"),
             func_ndim=3)
    subjects, _ = _generate_preproc_pipeline(config_file)
    assert subjects[0].scratch == "/tmp/scratch/sub001"
