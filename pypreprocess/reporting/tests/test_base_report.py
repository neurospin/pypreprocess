from nose.tools import assert_true
from pypreprocess.reporting.base_reporter import _get_software_versions


def test_get_software_versions():
    reports_nipype = False
    for software in _get_software_versions():
        if "Nipype" in software:
            reports_nipype = True
    assert_true(reports_nipype)
