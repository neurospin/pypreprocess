"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

XXX TODO: Docu
"""

import sys
import time
import tempita
import os

with open("template_reports/subject_preproc_report.tmpl.html", 'r') as fd:
    _text = fd.read()

    SUBJECT_PREPROC_REPORT_HTML_TEMPLATE = tempita.HTMLTemplate(_text)


def lines2breaks(lines):
    if type(lines) is str:
        lines = lines.split('\n')

    return tempita.HTMLTemplate("<br>".join(lines)).substitute()


with open("template_reports/dataset_preproc_report.tmpl.html") as fd:
    _text = fd.read()

    DATASET_PREPROC_REPORT_HTML_TEMPLATE = tempita.HTMLTemplate(_text)


def nipype2htmlreport(nipype_report_filename):
    with open(nipype_report_filename, 'r') as fd:
        return lines2breaks(fd.readlines())
