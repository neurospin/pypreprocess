import os
import pylab as pl
from nipy.labs import viz
import nibabel
import numpy as np
import nipy.labs.statistical_mapping as sm
from nipy.modalities.fmri.design_matrix import DesignMatrix
from base_reporter import *
import shutil


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
    mask: image object or string
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

    # sanity
    if isinstance(zmap, basestring):
        zmap = nibabel.load(zmap)
    if isinstance(mask, basestring):
        mask = nibabel.load(mask)

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


def generate_subject_stats_report(
    stats_report_filename,
    contrasts,
    z_maps,
    mask,
    design_matrix=None,
    subject_id=None,
    anat=None,
    anat_affine=None,
    threshold=2.3,
    cluster_th=0,
    start_time=None,
    progress_logger=None,
    shutdown_all_reloaders=True,
    **glm_kwargs
    ):
    """Generates a report summarizing the statistical methods and results

    Parameters
    ----------
    stats_report_filename: string:
        html file to which output (generated html) will be written

    contrasts: dict of arrays
        contrasts we are interested in; same number of contrasts as zmaps;
        same keys

    zmaps: dict of image objects or strings (image filenames)
        zmaps for contrasts we are interested in; one per contrast id

    mask: 'nifti image object'
        brain mask for ROI

    design_matrix: 'DesignMatrix' or np.lib.io.Npz containing such, or list
    of these
        design matrix(ces) for the experimental conditions

    contrasts: dict of arrays
       dictionary of contrasts of interest; the keys are the contrast ids,
       the values are contrast values (lists)

    z_maps: dict of 3D image objects or strings (image filenames)
       dict with same keys as 'contrasts'; the values are paths of z-maps
       for the respective contrasts

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

    shutdown_all_reloaders: bool (optional, default True)
        if True, all pages connected to the stats report page will
        be prevented from reloading after the stats report page
        has been completely generated

    **glm_kwargs:
        kwargs used to specify the control parameters used to specify the
        experimental paradigm and the GLM

    """

    # prepare for stats reporting
    if progress_logger is None:
        progress_logger = ProgressReport()

    output_dir = os.path.dirname(stats_report_filename)

    # copy css and js stuff to output dir
    shutil.copy(os.path.join(ROOT_DIR, "css/fsl.css"), output_dir)
    shutil.copy(os.path.join(ROOT_DIR, "js/jquery.min.js"), output_dir)
    shutil.copy(os.path.join(ROOT_DIR, "images/logo.jpeg"),
                output_dir)
    shutil.copy(os.path.join(ROOT_DIR, "images/failed.png"),
                output_dir)

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

    if len(glm_kwargs):
        def make_li(stuff):
            if isinstance(stuff, dict):
                val = "<ul>"
                for _k, _v in stuff.iteritems():
                    val += "<li>%s: %s</li>" % (_k, _v)
                val += "</ul>"
            else:
                val = str(stuff)

            return val

        methods += ("<p>The following control parameters were used for  "
                    " specifying the experimental paradigm and fitting the "
                    "GLM:<br/><ul>")
        for k, v in glm_kwargs.iteritems():
            methods += "<li>%s: %s</li>" % (k, make_li(v))
        methods += "</ul></p>"

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

    progress_logger.log("<b>Level 1 statistics</b><br/><br/>")

    # create design matrix thumbs
    design_matrices = design_matrix
    if not design_matrices is None:
        if not isinstance(design_matrices, list):
            design_matrices = [design_matrices]

        for design_matrix, j in zip(design_matrices,
                                    xrange(len(design_matrices))):
            if not isinstance(design_matrix, DesignMatrix):
                if design_matrix.endswith('.npz'):
                    npz = np.load(design_matrix)
                    design_matrix = DesignMatrix(npz['X'],
                                                 npz['conditions'],
                                                 )
                else:
                    raise TypeError(("Design matric format unknown; "
                                     "must be DesignMatrix object or"
                                     " .npz file containing such object"),
                                    )
            ax = design_matrix.show()
            ax.set_position([.05, .25, .9, .65])
            dmat_outfile = os.path.join(output_dir,
                                        'design_matrix_%i.png' % (j + 1),
                                        )
            pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)
            thumb = Thumbnail()
            thumb.a = a(href=os.path.basename(dmat_outfile))
            thumb.img = img(src=os.path.basename(dmat_outfile),
                                     height="500px",
                                     )
            thumb.description = "Design Matrix %i" % (j + 1)
            design_thumbs.commit_thumbnails(thumb)

    _vmax = 0
    _vmin = threshold
    for j in xrange(len(contrasts)):
        contrast_id = contrasts.keys()[j]
        contrast_val = contrasts[contrast_id]
        z_map = z_maps[contrast_id]

        # compute cut_coords for viz.plot_map(..) API
        # XXX review computation of cut_coords, vmin, and vmax; not clean!!!
        if isinstance(z_map, basestring):
            z_map = nibabel.load(z_map)
        pos_data = z_map.get_data() * (np.abs(z_map.get_data()) > 0)
        n_axials = 12
        delta_z_axis = 3
        z_axis_max = np.unravel_index(
            pos_data.argmax(), z_map.shape)[2]
        z_axis_min = np.unravel_index(
            np.argmax(-pos_data), z_map.shape)[2]
        z_axis_min, z_axis_max = (min(z_axis_min, z_axis_max),
                                  max(z_axis_max, z_axis_min))
        z_axis_min = min(z_axis_min, z_axis_max - delta_z_axis * n_axials)
        cut_coords = np.linspace(z_axis_min, z_axis_max, n_axials)

        # compute vmin and vmax
        vmax = pos_data.max()
        vmin = -vmax

        # vmax = max(- z_map.get_data().min(), z_map.get_data().max())
        # vmin = - vmax

        # # update colorbar endpoints
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

        title = z_map if isinstance(z_map, basestring) else None
        generate_level1_report(
            z_map, mask,
            stats_table,
            title=title,
            cluster_th=cluster_th,
            )

    # make colorbar for activations
    colorbar_outfile = os.path.join(output_dir,
                                    'activation_colorbar.png')
    make_standalone_colorbar(_vmin, _vmax, colorbar_outfile)

    # we're done, shut down re-loaders
    progress_logger.log('<hr/>')

    # prevent stats report page from reloading henceforth
    progress_logger.finish(stats_report_filename)

    # prevent any related page from reloading
    if shutdown_all_reloaders:
        progress_logger.finish_all()

    # return generated html
    with open(stats_report_filename, 'r') as fd:
        stats_report = fd.read()
        fd.close()

        return stats_report
