"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

XXX TODO: Document this module.
"""

import external.tempita.tempita as tempita
import sys
import time
import os
import shutil
import commands
import re


def del_empty_dirs(s_dir):
    """
    Recursively deletes all empty subdirs fo given dir.

    Parameters
    ==========
    s_dir: string
    directory under inspection

    """
    b_empty = True
    for s_target in os.listdir(s_dir):
        s_path = os.path.join(s_dir, s_target)
        if os.path.isdir(s_path):
            if not del_empty_dirs(s_path):
                b_empty = False
        else:
            b_empty = False
        if b_empty:
            print('deleting: %s' % s_dir)
            shutil.rmtree(s_dir)

    return b_empty


def export_report(src, make_archive=True):
    """
    Exports a report (html, php, etc. files) , ignoring data
    files like *.nii, etc.

    Parameters
    ==========
    src: string
    directory contain report

    make_archive: bool (optional)
    should the final report dir (dst) be archived ?

    """

    def check_extension(f):
        return re.match(".*\.(?:png|html|php|css)$", f)

    def ignore_these(folder, files):
        return [f for f in files if \
                    (os.path.isfile(
                    os.path.join(folder, f)) and not check_extension(f))]

    # sanity
    dst = os.path.join(src, "frozen_report")

    if os.path.exists(dst):
        print "Removing old %s." % dst
        shutil.rmtree(dst)

    # copy hierarchy
    print "Copying files directory structure from %s to %s" % (src, dst)
    shutil.copytree(src, dst, ignore=ignore_these)
    print "+++++++Done."

    # zip the results (dst)
    if make_archive:
        dst_archive = dst + ".zip"
        print "Writing archive %s .." % dst_archive
        print commands.getoutput(
            'cd %s; zip -r %s %s; cd -' % (os.path.dirname(dst),
                                           os.path.basename(dst_archive),
                                           os.path.basename(dst)))
        print "+++++++Done."


def GALLERY_HTML_MARKUP():
    """
    Function to generate markup for the contents of a <div id="results">
    type html element.

    """

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
    """
    HTML anchor element.

    """

    pass


class img(tempita.bunch):
    """
    HTML image element.

    """

    pass


class Thumbnail(tempita.bunch):
    """
    Thumbnnail (HTML img + effects).

    """

    pass


class ResultsGallery(object):
    """
    Gallery of results (summarized by thumbnails).

    """

    def __init__(self, loader_filename,
                 refresh_timeout=60000,  # reload every minute
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

        # touch loader file
        fd = open(self.loader_filename, 'a')
        fd.close()

    def commit_results_from_filename(self, filename):
        with open(filename) as fd:
            divs = fd.read()
            fd.close()

            loader_fd = open(self.loader_filename, 'a')
            loader_fd.write(divs)
            loader_fd.close()

    def commit_thumbnails(self, thumbnails):
        if not type(thumbnails) is list:
            thumbnails = [thumbnails]

        self.raw = GALLERY_HTML_MARKUP().substitute(thumbnails=thumbnails)

        fd = open(self.loader_filename, 'a')
        fd.write(self.raw)
        fd.close()


def SUBJECT_PREPROC_REPORT_HTML_TEMPLATE():
    """
    Report template for subject preproc.

    """

    with open(
        "template_reports/subject_preproc_report_template.tmpl.html") as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def DATASET_PREPROC_REPORT_HTML_TEMPLATE():
    """
    Returns report template for dataset preproc.

    """
    with open(
        "template_reports/dataset_preproc_report_template.tmpl.html") as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def lines2breaks(lines):
    """
    Converts line breaks to HTML breaks.

    """

    if type(lines) is str:
        lines = lines.split('\n')

    log = "<br>".join(lines)

    return tempita.HTMLTemplate(log).content


def nipype2htmlreport(nipype_report_filename):
    """
    Converts a nipype.caching report (.rst) to html.

    """
    with open(nipype_report_filename, 'r') as fd:
        return lines2breaks(fd.readlines())


#######
# DEMO
#######
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
