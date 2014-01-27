def test_bug_49_index_error_fix():
    from _test_utils import _make_sd
    from pypreprocess.reporting.preproc_reporter import generate_stc_thumbnails

    sd = _make_sd(ext=".nii.gz", n_sessions=2,
                  unique_func_names=True)
    sd.sanitize()
    sd.init_report()
    generate_stc_thumbnails(sd.func, sd.func,
                            sd.reports_output_dir)
