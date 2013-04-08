import os
import re
import time
import matplotlib as mpl
import pylab as pl
import external.tempita.tempita as tempita

# find package path
ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]

"""MISC"""
NIPY_URL = "http://nipy.sourceforge.net/nipy/stable/index.html"
SPM8_URL = "http://www.fil.ion.ucl.ac.uk/spm/software/spm8/"
PYPREPROCESS_URL = "https://github.com/neurospin/pypreprocess"
DARTEL_URL = ("http://www.fil.ion.ucl.ac.uk/spm/software/spm8/"
              "SPM8_Release_Notes.pdf")
NIPYPE_URL = "http://nipy.sourceforge.net/nipype/"


def lines2breaks(lines, delimiter="\n", number_lines=False):
    """
    Converts line breaks to HTML breaks, adding `pre` tags as necessary.

    Parameters
    ----------
    lines: string delimited by delimiter, or else list of strings
        lines to format into HTML format

    delimiter: string (default '\n')
        new-line delimiter, can be (escape) characters like '\n', '\r',
        '\r\n', '\t', etc.

    number_lines: boolean (default False)
        if false, then output will be line-numbered

    Returns
    -------
    HTML-formatted string

    """

    if isinstance(lines, basestring):
        lines = lines.split(delimiter)

    n_lines = len(lines)

    if not number_lines:
            lines = ["<pre>%s</pre>" % line for line in lines]
    else:
        lines = ["%i.\t<pre>%s</pre>" % (linum + 1, line)
                 for linum, line in zip(xrange(n_lines), lines)]

    log = "<br>".join(lines)

    return tempita.HTMLTemplate(log).content


def get_module_source_code(mod):
    """Function retrieved the source code of a module

    Parameters
    ----------

    mod: string or `module` handle
        existing filename of module, module handle

    Returns
    -------
    string, line-numbered HTML-formated code-block

    """

    if isinstance(mod, basestring):
        filename = mod
    elif isinstance(mod, type(os)):
        filename = mod.__file__

    with open(filename, 'r') as fd:
        lines = fd.read()

        return lines2breaks(lines, number_lines=True)


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
                 refresh_timeout=30,  # time between successive refreshs
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

    def commit_thumbnails(self, thumbnails, id=None):
        if not type(thumbnails) is list:
            thumbnails = [thumbnails]

        self.raw = GALLERY_HTML_MARKUP().substitute(thumbnails=thumbnails)

        fd = open(self.loader_filename, 'a')
        fd.write(self.raw)
        fd.close()


def commit_subject_thumnbail_to_parent_gallery(
    thumbnail,
    subject_id,
    parent_results_gallery):
    """Commit thumbnail (summary of subject_report) to parent results gallery,
    correcting attrs of the embedded img object as necessary.

    Parameters
    ----------
    thumbnail: base_reporter.Thumbnail instance
        thumbnail to be committed

    subject_id: string
        subject_id for subject under inspection

    result_gallery: base_reporter.ResultsGallery instance (optional)
        gallery to which thumbnail will be committed

    """

    if not thumbnail.img.src is None:
        thumbnail.img.height = "250px"
        thumbnail.img.src = "%s/%s" % (
            subject_id,
            os.path.basename(thumbnail.img.src))
        thumbnail.a.href = "%s/%s" % (
            subject_id,
            os.path.basename(thumbnail.a.href))
        parent_results_gallery.commit_thumbnails(thumbnail)


class ProgressReport(object):

    def __init__(self, report_filename=None, other_watched_files=[]):
        self.report_filename = report_filename
        self.other_watched_files = other_watched_files

        if not self.report_filename is None:
            open(self.report_filename, 'a').close()

    def log(self, msg):
        """Logs an html-formated stub to the report file

        Parameters
        ----------
        msg: string
            message to log

        """

        if self.report_filename is None:
            return

        with open(self.report_filename, 'r') as i_fd:
            content = i_fd.read()
            i_fd.close()
            marker = '<!-- log_next_thing_here -->'
            content = content.replace(marker, msg + marker)
            if not self.report_filename is None:
                with open(self.report_filename, 'w') as o_fd:
                    o_fd.write(content)
                    o_fd.close()

    def finish(self, report_filename=None):
        """Stops the automatic reloading (by the browser, etc.) of a given
         report page

         Parameters
         ----------
         report_filename: string (optinal)
             file URL of page to stop re-loading

        """

        if report_filename is None:
            report_filename = self.report_filename

        if not report_filename is None:
            with open(report_filename, 'r') as i_fd:
                content = i_fd.read()
                i_fd.close()

                # prevent pages from reloaded automaticall henceforth
                meta_reloader = "<meta http\-equiv=refresh content=.+?>"
                content = re.sub(meta_reloader, "", content)

                old_state = ("<font color=red><i>STILL RUNNING .."
                             "</i><blink>.</blink></font>")
                new_state = "Ended: %s" % time.ctime()
                new_content = content.replace(old_state, new_state)
                with open(report_filename, 'w') as o_fd:
                    o_fd.write(new_content)
                    o_fd.close()

    def finish_all(self, filenames=[]):
        """Stops the automatic re-loading of watched pages

        Parameters
        ----------
        filenames: string or list of strings
            filename(s) pointing to page(s) to stop automatic releading

        Examples
        --------
        >>> import os, glob, reporting.reporter as reporter
        >>> progress_logger = reporter.ProgressReport()
        >>> progress_logger.finish_all(glob.glob("/tmp/report*.html"))

        """

        for filename in [
            self.report_filename] + self.other_watched_files + filenames:
            self.finish(filename)

    def watch_file(self, filename):
        self.other_watched_files.append(filename)


def make_standalone_colorbar(vmin, vmax, colorbar_outfile=None):
    """Plots a stand-alone colorbar

    XXX Document this API!!!
    """

    fig = pl.figure(figsize=(6, 1))
    ax = fig.add_axes([0.05, 0.4, 0.9, 0.5])

    cmap = pl.cm.hot
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
    pl.savefig(colorbar_outfile)

    return cb


def SUBJECT_PREPROC_REPORT_HTML_TEMPLATE():
    """
    Report template for subject preproc.

    """

    with open(os.path.join(ROOT_DIR, 'template_reports',
                           'subject_preproc_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def DATASET_PREPROC_REPORT_HTML_TEMPLATE():
    """
    Returns report template for dataset preproc.

    """
    with open(os.path.join(ROOT_DIR, 'template_reports',
                           'dataset_preproc_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_SUBJECT_REPORT_LOG_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_subject_report_log_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_SUBJECT_REPORT_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
                           'fsl_subject_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_SUBJECT_REPORT_PREPROC_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_subject_report_preproc_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_SUBJECT_REPORT_STATS_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_subject_report_stats_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_DATASET_REPORT_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
                           'fsl_dataset_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_DATASET_REPORT_PREPROC_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_dataset_report_preproc_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_DATASET_REPORT_STATS_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_dataset_report_stats_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def FSL_DATASET_REPORT_LOG_HTML_TEMPLATE():
    """

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'fsl_dataset_report_log_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)
