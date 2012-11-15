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

_text = """\
{{default report_name = "Report"}}
{{default height = 200}}
<script src="jquery_path" type="text/javascript" charset="utf-8"></script>
<link rel="stylesheet" href="prettyphoto_css_path" type="text/css" media="screen" charset="utf-8" />
<script src="prettyphoto_js_path" type="text/javascript" charset="utf-8"></script>

<h2>{{report_name}}: pypreproc run, {{now}}</h2>
{{for summary_plot, title, full_plot, redirect_url in plots_gallery}}
<a href="{{summary_plot}}" rel="prettyPhoto[pp_gal]" title="{{title}}"><img src="{{full_plot}}" height="{{height}}" alt='<a href="{{redirect_url}}">goto detailed report</a>' /></a>
{{endfor}}

<script type="text/javascript" charset="utf-8">
  $(document).ready(function(){
    $("a[rel^='prettyPhoto']").prettyPhoto({
    allow_resize: true, /* Resize the photos bigger than viewport. true/false */
    default_width: 900,
    social_tools: false /* Hey, let's be serious here --please! */
    });
  });
</script>\
"""

_jquery_path =  os.path.join(PRETTYPHOTO_DIR,
                             "js/jquery-1.6.1.min.js")
_prettyphoto_js_path = os.path.join(
    PRETTYPHOTO_DIR, "js/jquery.prettyPhoto.js")
if not os.path.exists(_prettyphoto_js_path):
    raise Exception, "PrettyPhoto jquery plugin not found!"

_text = _text.replace("jquery_path", _jquery_path)
_text = _text.replace("prettyphoto_js_path", _prettyphoto_js_path)
_text = _text.replace("prettyphoto_css_path",
                      os.path.join(
        PRETTYPHOTO_DIR, "css/prettyPhoto.css"))

BASE_PREPROC_REPORT_HTML_TEMPLATE = tempita.HTMLTemplate(_text)


def nipype2htmlreport(nipype_report_filename):
    with open(nipype_report_filename, 'r') as fd:
        return tempita.HTMLTemplate(''.join(
                ['<p>%s</p>' % line for line in fd.readlines()])).substitute()

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
