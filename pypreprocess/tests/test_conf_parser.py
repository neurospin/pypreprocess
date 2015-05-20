import os
from nose.tools import assert_true
from nilearn._utils.testing import assert_raises_regex
from ..conf_parser import _generate_preproc_pipeline


def _make_config(out_file, **kwargs):
    config = """
[config]
"""
    for k, v in kwargs.items(): config += "%s=%s\r\n" % (k, v)
    fd = open(out_file, "w")
    fd.write(config)
    return config


def test_obligatory_params_config():
    dataset_dir = "/tmp/data"
    output_dir = "/tmp/output"
    if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)
    config_file = os.path.join(dataset_dir, "empty.ini")
    _make_config(config_file)
    assert_raises_regex(ValueError, "dataset_dir not specified",
                        _generate_preproc_pipeline, config_file)

    _make_config(config_file, dataset_dir=dataset_dir)
    assert_raises_regex(ValueError, "output_dir not specified",
                        _generate_preproc_pipeline, config_file)

    # this should not give any errors
    _make_config(config_file, dataset_dir=dataset_dir, output_dir=output_dir,
                 session_1_func="fM00223/fM00223_*.img")
    _generate_preproc_pipeline(config_file)


def test_issue110():
    dataset_dir = "/tmp/data"
    output_dir = "/tmp/output"
    if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)
    config_file = os.path.join(dataset_dir, "empty.ini")
    _make_config(config_file, newsegment=True, dataset_dir=dataset_dir,
                 output_dir=output_dir)
    _, options = _generate_preproc_pipeline(config_file)
    assert_true(options["newsegment"])
