"""
:Module: reporter
:Synopsis: utility module for report generation business
:Author: dohmatob elvis dopgima

XXX TODO: Document this module.
"""

import sys
import os
import shutil
import commands
import re
import time

import matplotlib as mpl
import pylab as pl
import nibabel
import numpy as np
from nipy.labs import viz

sys.path.append("..")
import external.tempita.tempita as tempita

# find package path
ROOT_DIR = os.path.split(os.path.abspath(__file__))[0]

# extention of web-related files (increment this as we support more
# and more file extensions for web business)
WEBBY_EXTENSION_PATTERN = ".*\.(?:png|jpeg|html|php|css)$"

"""MISC"""
NIPY_URL = "http://nipy.sourceforge.net/nipy/stable/index.html"


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


def export_report(src, tag="", make_archive=True):
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
        return re.match(WEBBY_EXTENSION_PATTERN, f)

    def ignore_these(folder, files):
        return [f for f in files if \
                    (os.path.isfile(
                    os.path.join(folder, f)) and not check_extension(f))]

    # sanity
    dst = os.path.join(src, "frozen_report_%s" % tag)

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
                 refresh_timeout=10,  # seconds
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


def lines2breaks(lines):
    """
    Converts line breaks to HTML breaks.

    """

    if type(lines) is str:
        lines = lines.split('\n')

    log = "<br>".join(lines)

    return tempita.HTMLTemplate(log).content


class ProgressReport(object):

    def __init__(self, report_filename, other_watched_files=[]):
        self.report_filename = report_filename
        self.other_watched_files = other_watched_files

    def log(self, msg):
        """Logs an html-formated stub to the report file

        Parameters
        ----------
        msg: string
            message to log

        """

        with open(self.report_filename, 'r') as i_fd:
            content = i_fd.read()
            i_fd.close()
            marker = '<!-- log_next_thing_here -->'
            content = content.replace(marker, msg + marker)
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

    def finish_all(self):
        """Stops the automatic re-loading of watched pages

        """

        self.finish()

        for filename in self.other_watched_files:
            self.finish(filename)

    def watch_file(self, filename):
        self.other_watched_files.append(filename)


def nipype2htmlreport(nipype_report_filename):
    """
    Converts a nipype.caching report (.rst) to html.

    """
    with open(nipype_report_filename, 'r') as fd:
        return lines2breaks(fd.readlines())


def generate_level1_report(zmap, mask,
                           output_html_path,
                           title="level 1 stats",
                           threshold=0.001,
                           method='fpr', cluster_th=0, null_zmax='bonferroni',
                           null_smax=None, null_s=None, nmaxima=4,
                           cluster_pval=.05):
    """
    Parameters
    ----------
    zmap: image object
        z-map data image
    mask: image object
        brain mask defining ROI
    output_html_path, string,
                      path where the output html should be written
    threshold, float, optional
               (p-variate) frequentist threshold of the activation image
    method, string, optional
            to be chosen as height_control in
            nipy.labs.statistical_mapping
    cluster_th, scalar, optional,
             cluster size threshold
    null_zmax: optional,
               parameter for cluster level statistics (?)
    null_s: optional,
             parameter for cluster level statistics (?)
    nmaxima: optional,
             number of local maxima reported per supra-threshold cluster
    """
    import nipy.labs.statistical_mapping as sm

    # Compute cluster statistics
    nulls = {'zmax': null_zmax, 'smax': null_smax, 's': null_s}

    """
    if null_smax is not None:
        print "a"
        clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                          nulls=nulls)
        clusters = [c for c in clusters if c['cluster_pvalue']<cluster_pval]
    else:
        print "b"
        clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                          height_control=method.lower(),
                                          cluster_th=cluster_th, nulls=nulls)
    """
    clusters, info = sm.cluster_stats(zmap, mask, height_th=threshold,
                                      nulls=nulls, cluster_th=cluster_th,)
    if clusters is not None:
        clusters = [c for c in clusters if c['cluster_pvalue'] < cluster_pval]

    #if clusters == None or info == None:
    #    print "No results were written for %s" % zmap_file_path
    #    return
    if clusters == None:
        clusters = []

    # Make HTML page
    output = open(output_html_path, mode="w")
    output.write("<center>\n")
    output.write("<b>%s</b>\n" % title)
    output.write("<table border = 1>\n")
    output.write("<tr><th colspan=4> Voxel significance </th>\
    <th colspan=3> Coordinates in MNI referential</th>\
    <th>Cluster Size</th></tr>\n")
    output.write("<tr><th>p FWE corr<br>(Bonferroni)</th>\
    <th>p FDR corr</th><th>Z</th><th>p uncorr</th>")
    output.write("<th> x (mm) </th><th> y (mm) </th><th> z (mm) </th>\
    <th>(voxels)</th></tr>\n")

    for cluster in clusters:
        maxima = cluster['maxima']
        size = cluster['size']
        for j in range(min(len(maxima), nmaxima)):
            temp = ["%.3f" % cluster['fwer_pvalue'][j]]
            temp.append("%.3f" % cluster['fdr_pvalue'][j])
            temp.append("%.2f" % cluster['zscore'][j])
            temp.append("%.3f" % cluster['pvalue'][j])
            for it in range(3):
                temp.append("%.0f" % maxima[j][it])
            if j == 0:
                # Main local maximum
                temp.append('%i' % size)
                output.write('<tr><th align="center">' + '</th>\
                <th align="center">'.join(temp) + '</th></tr>')
            else:
                # Secondary local maxima
                output.write('<tr><td align="center">' + '</td>\
                <td align="center">'.join(temp) + '</td><td></td></tr>\n')

    nclust = len(clusters)
    nvox = sum([clusters[k]['size'] for k in range(nclust)])

    output.write("</table>\n")
    output.write("Number of voxels: %i<br>\n" % nvox)
    output.write("Number of clusters: %i<br>\n" % nclust)

    if info is not None:
        output.write("Threshold Z = %.2f (%s control at %.3f)<br>\n" \
                     % (info['threshold_z'], method, threshold))
        output.write("Cluster size threshold p<%s" % cluster_pval)
    else:
        output.write("Cluster size threshold = %i voxels" % cluster_th)

    output.write("</center>\n")
    output.close()


def make_standalone_colorbar(vmin, vmax, colorbar_outfile=None):
    """Plots a stand-alone colorbar

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


def generate_subject_stats_report(
    stats_report_filename,
    design_matrix,
    contrasts,
    z_maps,
    subject_id,
    mask,
    anat=None,
    anat_affine=None,
    threshold=2.3,
    cluster_th=0,
    start_time=None,
    progress_logger=None,
    ):
    """Generates a report summarizing the statistical methods and results

    Parameters
    ----------
    stats_report_filename: string:
        html file to which output (generated html) will be written

    design_matrix: 'nipy design matrix' object
        design matrix for experiment

    contrasts: dict
       dictionary of contrasts of interest; the keys are the contrast ids,
       the values are contrast values (lists)

    z_maps: dict
       dict with same keys as 'contrasts'; the values are paths of z-maps
       for the respective contrasts

    mask: 'nifti image object'
        brain mask for subject

    anat: 3D array (optional)
        brain image to serve bg unto which activation maps will be plotted;
        passed to viz.plot_map API

    anat_affine: 2D array (optional)
        affine data for the anat

    threshold: float (optional)
        threshold to be applied to activation maps voxel-wise

    cluster_th: int (optional)
        minimal voxel count for clusteres declared as 'activated'

    start_time: string (optiona)
        start time for the stats analysis (useful for the generated
        report page)

    progress_logger: ProgressLogger object (optional)
        handle for logging progress

    """

    # prepare for stats reporting
    output_dir = os.path.dirname(stats_report_filename)
    stats_report_filename = os.path.join(output_dir,
                                         "report_stats.html")

    if progress_logger:
        progress_logger.watch_file(stats_report_filename)

    design_thumbs = ResultsGallery(
        loader_filename=os.path.join(output_dir,
                                     "design.html")
        )
    activation_thumbs = ResultsGallery(
        loader_filename=os.path.join(output_dir,
                                     "activation.html")
        )

    methods = """
    GLM and inference have been done using <a href="%s">nipy</a>. Statistic \
    images have been thresholded at Z>%s voxel-level.
    """ % (NIPY_URL, threshold)

    if start_time is None:
        start_time = time.ctime()
    level1_html_markup = FSL_SUBJECT_REPORT_STATS_HTML_TEMPLATE(
        ).substitute(
        start_time=start_time,
        subject_id=subject_id,
        methods=methods)
    with open(stats_report_filename, 'w') as fd:
        fd.write(str(level1_html_markup))
        fd.close()

    if progress_logger:
        progress_logger.log("<b>Level 1 statistics</b><br/><br/>")

    # show design matrix
    ax = design_matrix.show()
    ax.set_position([.05, .25, .9, .65])
    dmat_outfile = os.path.join(output_dir, 'design_matrix.png')
    pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)
    thumb = Thumbnail()
    thumb.a = a(href=os.path.basename(dmat_outfile))
    thumb.img = img(src=os.path.basename(dmat_outfile),
                             height="400px",
                             )
    thumb.description = "Design Matrix"
    design_thumbs.commit_thumbnails(thumb)

    _vmax = 0
    _vmin = threshold
    for j in xrange(len(contrasts)):
        contrast_id = contrasts.keys()[j]
        contrast_val = contrasts[contrast_id]
        map_path = z_maps[contrast_id]
        z_map = nibabel.load(map_path)

        # get positive z_map
        pos_data = z_map.get_data() * (z_map.get_data() > 0)

        # compute cut_coords for viz.plot_map(..) API
        n_axials = 12
        delta_z_axis = 3
        z_axis_max = np.unravel_index(
            pos_data.argmax(), z_map.shape)[2]
        z_axis_min = np.unravel_index(
            pos_data.argmin(), z_map.shape)[2]
        z_axis_min, z_axis_max = (min(z_axis_min, z_axis_max),
                                  max(z_axis_max, z_axis_min))
        z_axis_min = min(z_axis_min, z_axis_max - delta_z_axis * n_axials)
        cut_coords = np.linspace(z_axis_min, z_axis_max, n_axials)

        # compute vmin and vmax
        vmax = pos_data.max()
        vmin = pos_data.min()

        # update colorbar endpoints
        _vmax = max(_vmax, vmax)

        # plot activation proper
        viz.plot_map(pos_data, z_map.get_affine(),
                     cmap=pl.cm.hot,
                     anat=anat,
                     anat_affine=anat_affine,
                     vmin=vmin,
                     vmax=vmax,
                     threshold=threshold,
                     slicer='z',
                     cut_coords=cut_coords,
                     black_bg=True,
                     )

        # store activation plot
        z_map_plot = os.path.join(output_dir,
                                  "%s_z_map.png" % contrast_id)
        pl.savefig(z_map_plot, dpi=200, bbox_inches='tight',
                   facecolor="k",
                   edgecolor="k")
        stats_table = os.path.join(output_dir,
                                   "%s_stats_table.html" % contrast_id)

        # create thumbnail for activation
        thumbnail = Thumbnail()
        thumbnail.a = a(href=os.path.basename(stats_table))
        thumbnail.img = img(
            src=os.path.basename(z_map_plot), height="250px",)
        thumbnail.description = "%s contrast: %s" % (contrast_id, contrast_val)
        activation_thumbs.commit_thumbnails(thumbnail)

        generate_level1_report(
            z_map, mask,
            stats_table,
            title=map_path,
            cluster_th=cluster_th,
            )

    # make colorbar for activations
    colorbar_outfile = os.path.join(output_dir,
                                    'activation_colorbar.png')
    make_standalone_colorbar(_vmin, _vmax, colorbar_outfile)

    # we're done, shut down all re-loaders
    if progress_logger:
        progress_logger.log('<hr/>')
        progress_logger.finish_all()

    # return generated html
    with open(stats_report_filename, 'r') as fd:
        stats_report = fd.read()
        fd.close()

        return stats_report
