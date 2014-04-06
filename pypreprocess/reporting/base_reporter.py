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
from ..external.tempita import (HTMLTemplate,
                        bunch
                        )

# find package path
ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]

# find package path
ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]

# MISC
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

    Parameters
    ----------
    lines: list or string
       linbes to convert to HTML breaks <br/>

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
        elif isinstance(stuff, list) or hasattr(stuff, "__iter__"):
            return '<ul type="none"><li>%s</li></ul>' % list(stuff)
        else:
            # XXX handle other bundled types which are not necessarily
            # dict-like!!!
            val = str(stuff)

        return val

    if isinstance(mydict, basestring):
        return mydict
    elif isinstance(mydict, list):
        return make_li(mydict)
    elif isinstance(mydict, dict):
        html_ul = ""
        for k, v in mydict.iteritems():
            if not v is None:
                html_ul += "<li>%s: %s</li>" % (k, make_li(v))
        html_ul += "</ul>"
    else:
        raise TypeError(
            "Input type must be string, list, or dict, got %s" % mydict)

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

    Examples
    --------
    >>> thumb = Thumbnail(description='sub001', a=a(href='https://github.com'),
    ... img=img(src='/tmp/logo.gif'))
    >>> gallery = get_gallery_html_markup().substitute(thumbnails=[thumb])
    >>> gallery
    <div class="img">\n  <a href="https://github.com">\n    <img \
    src="/tmp/logo.gif"/>\n  </a>\n  <div class="desc">sub001</div>\n</div>\n

    """

    return HTMLTemplate(
        """
{{for thumbnail in thumbnails}}
<div class="img">
  <a {{attr(**thumbnail.a)}}>
    <img {{attr(**thumbnail.img)}}/>
  </a>
  <div class="desc">{{thumbnail.description | html}}</div>
</div>
{{endfor}}
""")


class _HTMLElement(bunch):
    """
    Parameters
    ----------
    All parameters must be specified in 'param=value' style. These can be any
    HTML anchor parameter. Below we document only the compulsary paremters.

    Parameters
    ----------
    **kwargs: dict-like
        param-value dict of attributes for this HTML elemnt.

    """

    _compulsary_params = []

    def __init__(self, **kwargs):

        bunch.__init__(self, **kwargs)

        for param in self._compulsary_params:
            if not param in kwargs:
                raise ValueError(
                    "Need to specify '%s' parameter for HTML %s" % (
                        param, self.__class__.__name__))


class a(_HTMLElement):
    """
    HTML anchor element.

    Parameters
    ----------
    **kwargs: dict-like
        param-value dict of attributes for this HTML anchor element.

    Examples
    --------
    >>> a = a(href='http://gihub.com/neurospin/pypreprocess')
    >>> a
    <a href='http://gihub.com/neurospin/pypreprocess'>

    """

    # _compulsary_params = ['href']

    def __init__(self, **kwargs):
        _HTMLElement.__init__(self, **kwargs)


class img(_HTMLElement):
    """
    HTML image element.

    Parameters
    ----------
    **kwargs: dict-like
        param-value dict of attributes for this HTML image element. Compulsary
        parameter-values are:

        src: string
            src the image
        href: string
            href of the image

    Examples
    --------
    >>> img = img(src='logo.png')
    >>> img
    <img src='logo.png'>

    """

    # _compulsary_params = ['src']

    def __init__(self, **kwargs):
        _HTMLElement.__init__(self, **kwargs)


class Thumbnail(_HTMLElement):
    """
    HTML thumbnail.

    Parameters
    ----------
    **kwargs: dict-like
        param-value dict of attributes for this HTML image element. Comulsary
        parameter-values are:

        description: string
            description of the thumbnail
        a: a `object`
            HTML anchor `a` object for the thumbnail
        img: img `object`
            HTML image `img` object for the thumbnail

    Examples
    --------
    >>> thumb = Thumbnail(description='sub001', a=a(href='https://github.com'),
    ... img=img(src='/tmp/logo.gif'))
    >>> thumb
    <Thumbnail a=<a href='https://github.com'> description='sub001' \
    img=<img src='/tmp/logo.gif'>>

    """

    # _compulsary_params = ['description', 'a', 'img']

    def __init__(self, **kwargs):
        _HTMLElement.__init__(self, **kwargs)


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
        # sanitize thumbnail
        if not type(thumbnails) is list:
            thumbnails = [thumbnails]

        for thumbnail in thumbnails:
            thumbnail.description = dict_to_html_ul(thumbnail.description)

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
    thumbnail: Thumbnail instance
        thumbnail to be committed

    subject_id: string
        subject_id for subject under inspection

    result_gallery: ResultsGallery instance (optional)
        gallery to which thumbnail will be committed

    """

    # sanitize thumbnail
    assert hasattr(thumbnail, 'img')
    assert not thumbnail.img is None

    # resize thumbnail
    thumbnail.img.height = "250px"

    if thumbnail.img.src:
        thumbnail.img.src = os.path.join(subject_id, "reports",
                                         os.path.basename(thumbnail.img.src))
    if thumbnail.a.href:
        thumbnail.a.href = os.path.join(subject_id, "reports",
                                        os.path.basename(thumbnail.a.href))

    # commit thumbnail to parent's gallery
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

        with open(self.log_filename, 'a') as ofd:
            ofd.write(msg + "<br/>")

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
            new_state = "Ended: %s" % pretty_time()
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
    pl.close()

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


def _get_template(template_file, **kwargs):
    with open(template_file) as fd:
        _text = fd.read()
        fd.close()

        return HTMLTemplate(_text).substitute(**kwargs)


def get_subject_report_log_html_template(**kwargs):
    """Returns html template (string) for subject log report

    """

    return _get_template(os.path.join(
            ROOT_DIR, 'template_reports',
            'subject_report_log_template.tmpl.html'),
                         **kwargs)


def get_subject_report_html_template(**kwargs):
    """Returns html tamplate (string) for subject report (page de garde)

    """

    return _get_template(os.path.join(
            ROOT_DIR, 'template_reports',
            'subject_report_template.tmpl.html'),
                         **kwargs)


def get_subject_report_preproc_html_template(**kwargs):
    """Returns html template (string) for subject preproc report page

    """

    return _get_template(os.path.join(
            ROOT_DIR, 'template_reports',
            'subject_report_preproc_template.tmpl.html'),
                         **kwargs)


def get_subject_report_stats_html_template(**kwargs):
    """Returns html template (string) for subject stats report page

    """

    return _get_template(os.path.join(
                ROOT_DIR, 'template_reports',
                'subject_report_stats_template.tmpl.html'),
                         **kwargs)


def get_ica_html_template(**kwargs):
    """Returns html template (string) for subject stats report page

    """

    return _get_template(os.path.join(
                ROOT_DIR, 'template_reports',
                'ica_report_template.tmpl.html'),
                         **kwargs)


def get_dataset_report_html_template(**kwargs):
    """Returns html template (string) for dataset report page (page de garde)

    """

    return _get_template(os.path.join(ROOT_DIR, 'template_reports',
                                      'dataset_report_template.tmpl.html'),
                         **kwargs)


def get_dataset_report_preproc_html_template(**kwargs):
    """Returns html template (string) for dataset preproc report page

    """

    return _get_template(os.path.join(
            ROOT_DIR, 'template_reports',
            'dataset_report_preproc_template.tmpl.html'),
                         **kwargs)


def get_dataset_report_stats_html_template(**kwargs):
    """Returns html template (string) for dataset stats report page

    """

    return _get_template(os.path.join(
            ROOT_DIR, 'template_reports',
            'dataset_report_stats_template.tmpl.html'),
                         **kwargs)


def get_dataset_report_log_html_template(**kwargs):
    """Returns html template (string) for dataset log report page

    """

    return _get_template(os.path.join(
            ROOT_DIR, 'template_reports',
            'dataset_report_log_template.tmpl.html'),
                         **kwargs)


def copy_failed_png(output_dir):
    """
    Copies failed.png image to output_dir

    """

    shutil.copy(os.path.join(ROOT_DIR, "images/failed.png"),
                output_dir)


def copy_web_conf_files(output_dir):
    """Function to copy css, js, icon files to given directory.

    """

    def _copy_web_conf_file_ext(src_dir_basename, extentions, ignore=None):
        if ignore is None:
            ignore = []
        if isinstance(ignore, basestring):
            ignore = [ignore]

        for ext in extentions:
            for src in glob.glob(os.path.join(ROOT_DIR, "%s/*%s" % (
                        src_dir_basename, ext))):

                # skip faild.png image (see issue #30)
                if src.endswith("failed.png"):
                    continue

                shutil.copy(src, output_dir)

    # copy js stuff
    _copy_web_conf_file_ext("js", ['.js'])

    # copy css stuf
    _copy_web_conf_file_ext("css", ['.css'])

    # copy icons
    _copy_web_conf_file_ext("icons", ['.jpg', '.jpeg', '.png', '.gif'])

    # copy images
    _copy_web_conf_file_ext("images", ['.jpg', '.jpeg', '.png', '.gif'])


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


def pretty_time():
    """
    Returns currenct time in the format: hh:mm:ss ddd mmm yyyy

    """

    return " ".join([time.ctime().split(" ")[i] for i in [3, 0, 2, 1, 4]])
