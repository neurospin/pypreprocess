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

PRETTYPHOTO_DIR = os.getcwd()
if 'PRETTYPHOTO_DIR' in os.environ:
    PRETTYPHOTO_DIR = os.environ['PRETTYPHOTO_DIR']

_jquery_path = os.path.join(PRETTYPHOTO_DIR,
                             "js/jquery-1.6.1.min.js")
_prettyphoto_js_path = os.path.join(
    PRETTYPHOTO_DIR, "js/jquery.prettyPhoto.js")
if not os.path.exists(_prettyphoto_js_path):
    raise Exception("PrettyPhoto jquery plugin not found!")

with open("reporter.tmpl.html", 'r') as fd:
    _text = fd.read()
    _text = _text.replace("jquery_path", _jquery_path)
    _text = _text.replace("prettyphoto_js_path", _prettyphoto_js_path)
    _text = _text.replace("prettyphoto_css_path",
                          os.path.join(
            PRETTYPHOTO_DIR, "css/prettyPhoto.css"))

    BASE_PREPROC_REPORT_HTML_TEMPLATE = tempita.HTMLTemplate(_text)

with open("subject_realignment_report.tmpl.html", 'r') as fd:
    _text = fd.read()
    _text = _text.replace("jquery_path", _jquery_path)
    _text = _text.replace("prettyphoto_js_path", _prettyphoto_js_path)
    _text = _text.replace("prettyphoto_css_path",
                          os.path.join(
            PRETTYPHOTO_DIR, "css/prettyPhoto.css"))

    SUBJECT_REALIGNMENT_REPORT_HTML_TEMPLATE = tempita.HTMLTemplate(_text)

with open("subject_preproc_report.tmpl.html", 'r') as fd:
    _text = fd.read()
    _text = _text.replace("jquery_path", _jquery_path)
    _text = _text.replace("prettyphoto_js_path", _prettyphoto_js_path)
    _text = _text.replace("prettyphoto_css_path",
                          os.path.join(
            PRETTYPHOTO_DIR, "css/prettyPhoto.css"))

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

if __name__ == '__main__':
    tmpl = BASE_PREPROC_REPORT_HTML_TEMPLATE
    now = time.ctime()
    plots_gallery = [(x.replace('_summary.png', '.png'),
                      '[place title here]', x, "http://github.com/dohmatob") \
                         for x in sys.stdin.readlines()]

    report = tmpl.substitute(locals())

    print report

    with open("demo.html", 'w') as fd:
        fd.write(report)
        fd.close()
