import reporting.base_reporter as base_reporter

dataset_description = "openfmri rules"
loader_filename = '/tmp/openfmri_loader.php'
report_filename = '/tmp/report_preproc.html'
preproc_undergone = "unspecified"
parent_results_gallery = base_reporter.ResultsGallery(
    loader_filename=loader_filename,
    refresh_timeout=30,
    )


preproc_params = {'today': 'yes', 'toto': 1}

preproc = base_reporter.get_openfmri_html_report_template(
    ).substitute(
    results=None,  # parent_results_gallery,
    # start_time=time.ctime(),
    preproc_undergone=preproc_undergone,
    dataset_description=dataset_description,
    # source_code=user_source_code,
    # source_script_name=user_script_name,
    preproc_params=preproc_params,
    )

# dump report
with open(report_filename, 'w') as fd:
    fd.write(str(preproc))
    fd.close()
