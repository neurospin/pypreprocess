"""
:Module: base_reporter
:Synopsis: basic utilities (functions, classes) for the reporting business
:Author: dohmatob elvis dopgima

XXX Write test(case)s for this module

"""

import os
import re
import glob
import shutil
import time
import matplotlib as mpl
import pylab as pl
import numpy as np
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

    if not number_lines:
        lines = ["%s" % line for line in lines]
        output = "<pre>%s</pre>" % "".join(lines)
    else:
        lines = ["<li>%s</li>" % line for line in lines]
        output = "<ol><pre>" + "".join(lines) + "</pre></ol>"

    return output


def dict_to_html_ul(mydict):
    """Function converts dict to an HTML ul element

    Parameters
    ----------
    mydict: dict-like object
        dict (could be dict of any things), perhaps of dicts) to
        convert to HTML ul

    Returns
    -------
    String, ul element

    """

    html_ul = ""

    def make_li(stuff):
        # handle dict type
        if isinstance(stuff, dict):
            val = "<ul>"
            for _k, _v in stuff.iteritems():
                if not _v is None:
                    val += "<li>%s: %s</li>" % (_k, make_li(_v))
            val += "</ul>"
        # handle tuple type
        elif isinstance(stuff, tuple):
            return '<ul type="none"><li>%s</li></ul>' % tuple(stuff)
        # handle array-like type type
        elif isinstance(stuff, list) or hasattr(stuff, "tolist"):
            return '<ul type="none"><li>%s</li></ul>' % list(stuff)
        else:
            # XXX handle other bundled types which are not necessarily
            # dict-like!!!
            val = str(stuff)

        return val

    for k, v in mydict.iteritems():
        if not v is None:
            html_ul += "<li>%s: %s</li>" % (k, make_li(v))
    html_ul += "</ul>"

    return html_ul


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


def get_gallery_html_markup():
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

        self.raw = get_gallery_html_markup().substitute(thumbnails=thumbnails)

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
    """Class encapsulating functionality for logging arbitrary html stubs
    to report pages, modifying pages dynamically (like disabling automatic
    releads, etc.)

    """

    def __init__(self, log_filename=None, other_watched_files=[]):
        """Constructor

        Parameters
        ----------
        log_filename: string (optional, default None)
            filename to which html stubs will be logged

         other_watched_files: list or similar iterable (optional, default [])
            files watched by the progress reporter; for example whenever the
            `finish_all` method is invoked, all watched files are disabled for
            automatic reloading thenceforth

         """

        self.log_filename = log_filename

        if not self.log_filename is None:
            open(self.log_filename, 'a').close()

        self.watched_files = []
        self.watch_files(other_watched_files)

    def watch_files(self, filenames):
        for filename in filenames:
            self.watch_file(filename)

    def log(self, msg):
        """Logs an html-formated stub to the report file

        Parameters
        ----------
        msg: string
            message to log

        """

        if self.log_filename is None:
            return

        with open(self.log_filename, 'r') as i_fd:
            content = i_fd.read()
            i_fd.close()
            marker = '<!-- log_next_thing_here -->'
            content = content.replace(marker, msg + marker)
            if not self.log_filename is None:
                with open(self.log_filename, 'w') as o_fd:
                    o_fd.write(content)
                    o_fd.close()

    def finish(self, filename):
        """Stops the automatic reloading (by the browser, etc.) of a given
         report page

         Parameters
         ----------
         filename:
             file URL of page to stop re-loading

        """

        with open(filename, 'r') as i_fd:
            content = i_fd.read()
            i_fd.close()

            # prevent pages from reloaded automaticall henceforth
            meta_reloader = "<meta http\-equiv=refresh content=.+?>"
            content = re.sub(meta_reloader, "", content)

            old_state = ("<font color=red><i>STILL RUNNING .."
                         "</i><blink>.</blink></font>")
            new_state = "Ended: %s" % time.ctime()
            new_content = content.replace(old_state, new_state)
            with open(filename, 'w') as o_fd:
                o_fd.write(new_content)
                o_fd.close()

    def _finish_files(self, filenames):
        for filename in filenames:
            self.finish(filename)

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

        self._finish_files(self.watched_files + filenames)

    def finish_dir(self, dirname, filename_wildcat="*.html"):
        """'Finishes' all pages (html, etc., files) in a given directory.

        Parameters
        ----------
        dirname: string
            name of existing directory to 'finish'

        filename: string (optional, default "*.html")
            wildcat defining files to 'finish' (useful for globbing) in dirname

        """

        self._finish_files(
            glob.glob(os.path.join(dirname, filename_wildcat)))

    def watch_file(self, filename):
        """Specifies (yet another) file to be watched.

        Parameters
        ----------
        filename: string
            existing filename

        """

        assert isinstance(filename, basestring)
        self.watched_files.append(filename)


def make_standalone_colorbar(cmap, vmin, vmax, colorbar_outfile=None):
    """Plots a stand-alone colorbar

    Parameters
    ----------
    cmap: some colormap object (like pylab.cm.hot, et
    nipy.lab.viz.cold_hot, etc.)
        colormap to use in plotting colorbar
    vmin: float
        param passed to `mpl.colors.Normalize`

    vmax: float
        param passed to `mpl.colors.Normalize`

    colorbar_outfil: string (optional, default None)
        outputfile for plotted colorbar

    """

    vmin, vmax = min(vmin, vmax), max(vmin, vmax)

    fig = pl.figure(figsize=(6, 1))
    ax = fig.add_axes([0.05, 0.4, 0.9, 0.5])

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')

    # save colorbar
    pl.savefig(colorbar_outfile, bbox_inches='tight')

    return cb


def get_cut_coords(map3d, n_axials=12, delta_z_axis=3):
    """
    Heuristically computes optimal cut_coords for plot_map(...) call.

    Parameters
    ----------
    map3d: 3D array
        the data under consideration
    n_axials: int, optional (default 12)
        number of axials in the plot
    delta_z_axis: int, optional (default 3)
        z-axis spacing

    Returns
    -------
    cut_coords: 1D array of length n_axials
        the computed cut_coords

    """

    z_axis_max = np.unravel_index(
        np.abs(map3d).argmax(), map3d.shape)[2]
    z_axis_min = np.unravel_index(
        (-np.abs(map3d)).argmin(), map3d.shape)[2]
    z_axis_min, z_axis_max = (min(z_axis_min, z_axis_max),
                              max(z_axis_max, z_axis_min))
    z_axis_min = min(z_axis_min, z_axis_max - delta_z_axis * n_axials)

    cut_coords = np.linspace(z_axis_min, z_axis_max, n_axials)

    return cut_coords


def compute_vmin_vmax(map3d):
    """
    Computes vmin and vmax params for plot_map.

    """

    vmax = max(map3d.max(), -map3d.min())
    vmin = -vmax

    return vmin, vmax


def get_subject_report_log_html_template():
    """Returns html template (string) for subject log report

    """

    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'subject_report_log_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def get_subject_report_html_template():
    """Returns html tamplate (string) for subject report (page de garde)

    """

    with open(os.path.join(
            ROOT_DIR, 'template_reports',
                           'subject_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def get_subject_report_preproc_html_template():
    """Returns html template (string) for subject preproc report page

    """

    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'subject_report_preproc_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def get_subject_report_stats_html_template():
    """Returns html template (string) for subject stats report page

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'subject_report_stats_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def get_ica_html_template():
    """Returns html template (string) for subject stats report page

    """
    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'ica_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def get_dataset_report_html_template():
    """Returns html template (string) for dataset report page (page de garde)

    """

    with open(os.path.join(
            ROOT_DIR, 'template_reports',
                           'dataset_report_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def get_dataset_report_preproc_html_template():
    """Returns html template (string) for dataset preproc report page

    """

    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'dataset_report_preproc_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def get_dataset_report_stats_html_template():
    """Returns html template (string) for dataset stats report page

    """

    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'dataset_report_stats_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def get_dataset_report_log_html_template():
    """Returns html template (string) for dataset log report page

    """

    with open(os.path.join(
            ROOT_DIR, 'template_reports',
            'dataset_report_log_template.tmpl.html')) as fd:
        _text = fd.read()
        return tempita.HTMLTemplate(_text)


def copy_web_conf_files(output_dir):
    """Function to copy css, js, icon files to given directory.

    """

    for js_file in glob.glob(os.path.join(ROOT_DIR,
                                          "js/*.js")):
        shutil.copy(js_file, output_dir)
    for css_file in glob.glob(os.path.join(ROOT_DIR,
                                          "css/*.css")):
        shutil.copy(css_file, output_dir)
    for icon_file in glob.glob(os.path.join(ROOT_DIR,
                                          "icons/*.gif")):
        shutil.copy(icon_file, output_dir)


def copy_report_files(src, dst):
    """Backs-up report files (*.html, *.js, etc.) from src to dst

    """

    if not os.path.exists(dst):
        os.makedirs(dst)

    for ext in ["css", "html", "js", "php", "png", "jpeg",
                "jpg", "gif", "json"]:
        for x in glob.glob(os.path.join(src,
                                        "*.%s" % ext)):
            shutil.copy(x, dst)
