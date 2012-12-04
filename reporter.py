"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

XXX TODO: Docu
"""

import tempita
import sys
import time
import os


def GALLERY_HTML_MARKUP():
    return tempita.HTMLTemplate("""\
{{for thumbnail in thumbnails}}
<div class="img">
  <a {{attr(**thumbnail.a)}}>
    <img {{attr(**thumbnail.img)}}/>
  </a>
  <div class="desc">{{thumbnail.description | html}}</div>
</div>
{{endfor}}""")

class a(tempita.bunch):
    pass


class img(tempita.bunch):
    pass


class Thumbnail(object):
    pass


class ResultsGallery(object):

    def __init__(self, loader_filename,
                 refresh_timeout=5000,
                 title='Results',
                 description=None
                 ):
        self.loader_filename = loader_filename
        self.refresh_timeout = refresh_timeout
        self.title = title
        self.description = description

        # start with a clean slate
        if os.path.isfile(self.loader_filename):
            os.remove(self.loader_filename)

    def commit_thumbnails(self, thumbnails):
        if not type(thumbnails) is list:
            thumbnails = [thumbnails]

        self.raw = GALLERY_HTML_MARKUP().substitute(thumbnails=thumbnails)

        fd = open(self.loader_filename, 'a')
        fd.write(self.raw)
        fd.close()


def SUBJECT_PREPROC_REPORT_HTML_TEMPLATE():
    with open("template_reports/subject_preproc_report_template.tmpl.html") as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def DATASET_PREPROC_REPORT_HTML_TEMPLATE():
    with open(
        "template_reports/dataset_preproc_report_template.tmpl.html") as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def lines2breaks(lines):
    if type(lines) is str:
        lines = lines.split('\n')

    return tempita.HTMLTemplate("<br>".join(lines)).substitute()


def nipype2htmlreport(nipype_report_filename):
    with open(nipype_report_filename, 'r') as fd:
        return lines2breaks(fd.readlines())


if __name__ == '__main__':
    with open(("template_reports/dataset_preproc_report"
               "_template.tmpl.html")
              ) as fd:

        # read template
        tmpl = tempita.HTMLTemplate(fd.read())

        # dataset description (markup of)
        dataset_description = """\
<p>The NYU CSC TestRetest resource includes EPI-images of 25 participants \
gathered during rest as well as anonymized anatomical images of the same \
participants.</p>
<p>The resting-state fMRI images were collected on several occasions:<br>
1. the first resting-state scan in a scan session<br>
2. 5-11 months after the first resting-state scan<br>
3. about 30 (< 45) minutes after 2.</p>
<p>Get full description <a target="_blank" href="http://www.nitrc.org\
/projects/nyu_trt/">here</a>.</p>"""

        # generate html markup for gallery
        results_description = """\
<p>Below is a gallery of plots summarising the results of each preprocessed\
 subject. Hover over image to get tooltip; click to get detailed report for \
subject.</p>"""

        # file containing (dynamic) code for updating results
        loader_filename = '/tmp/results_loader.php'

        results = ResultsGallery(loader_filename,
                                 description=results_description)

        preproc_undergone = """\
<p>All preprocessing has been done using <a target='_blank' href='http://nipy\
.sourceforge.net/nipype/'>nipype</a>'s interface to the <a target='_blank' \
href='http://www.fil.ion.ucl.ac.uk/spm/'>SPM8 package</a>.</p>
<p>Only intra-subject preprocessing has been carried out. For each \
subject:<br/>
1. motion correction has been done so as to detect artefacts due to the \
subject's head motion during the acquisition, after which the images \
have been resliced;<br/>
2. the fMRI images (a 4D time-series made of 3D volumes aquired every TR \
seconds) have been coregistered against the subject's anatomical. At the \
end of this stage, the fMRI images have gained some anatomical detail and \
resolution;<br/>
3. the subject's anatomical has been segmented into GM, WM, and CSF tissue \
probabitility maps (TPMs);<br/>
4. the learned transformations have been used to normalize the coregistered \
fMRI images</p>"""

        timestamp = time.ctime().replace('  ', ' ')
        timestamp = timestamp.split(' ')
        timestamp = ' '.join(timestamp[3:4] + timestamp[:3] + timestamp[4:5])

        print tmpl.substitute(
            timestamp=timestamp,
            dataset_description=dataset_description,
            results=results,
            preproc_undergone=preproc_undergone)

        for plot in sys.stdin.readlines():
            subject_id = os.path.basename(
                os.path.dirname(plot))
            subject_report_page = os.path.join(os.path.dirname(plot),
                                               "_report.html")
            # create html anchor element
            a = tempita.bunch(href=subject_report_page)

            # create html img element
            img = tempita.bunch(src=plot,
                                height="250px",
                                alt="")

            # create thumbnail with given anchor an img
            thumbnail = Thumbnail()
            thumbnail.a = a
            thumbnail.img = img
            thumbnail.description = subject_id

            # update gallery
            results.commit_thumbnails(thumbnail)
