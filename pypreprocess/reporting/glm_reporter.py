import os
import sys
import numpy as np
import pylab as pl
import nibabel
from nilearn.plotting import plot_stat_map
from nilearn.image import reorder_img
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs.mask import intersect_masks
import base_reporter
from cluster_level_analysis import  cluster_stats


def generate_level1_stats_table(zmap, mask,
                                output_html_path,
                                p_threshold=.001,
                                z_threshold=None,
                                method='fpr',
                                cluster_th=15,
                                null_zmax='bonferroni',
                                null_smax=None,
                                null_s=None,
                                nmaxima=4,
                                cluster_pval=.05,
                                title=None,
                                ):
    """Function to generate level 1 stats table for a contrast.

    Parameters
    ----------
    zmap: image object
        z-map data image

    mask: image object or string


        brain mask defining ROI

    output_html_path: string,
        path where the output html should be written

    p_threshold: float (optional, default .001)
        (p-variate) frequentist threshold of the activation image

    z_threshold: float (optional, default None)
        Threshold that has been applied to Z map (input z_map)

    method: string (optional, default 'fpr')
        to be chosen as height_control in nipy.labs.statistical_mapping

    cluster_th: int (optional, default 15)
        cluster size threshold (in # voxels)

    null_zmax: string (optional, default 'bonferroni')
        parameter for cluster level statistics (?)

    null_s: strint (optional, default None)
        parameter for cluster level statistics (?)

    nmaxima: int (optional, default 4)
        number of local maxima reported per supra-threshold cluster

    title: string (optional)
        title of generated stats table

    """
    # sanity
    if isinstance(zmap, basestring):
        zmap = nibabel.load(zmap)
    if isinstance(mask, basestring):
        mask = nibabel.load(mask)

    # Compute cluster statistics
    nulls = {'zmax': null_zmax, 'smax': null_smax, 's': null_s}

    # do some sanity checks
    if title is None:
        title = "Level 1 Statistics"
    clusters, _ = cluster_stats(zmap, mask, height_th=p_threshold,
                                nulls=nulls, cluster_th=cluster_th)
    if clusters is not None:
        clusters = [c for c in clusters if c['cluster_p_value'] < cluster_pval]
    else:
        clusters = []

    # Make HTML page
    page = open(output_html_path, mode="w")
    page.write("<center>\n")
    page.write("<b>%s</b>\n" % title)
    page.write("<table border = 1>\n")
    page.write("<tr><th colspan=4> Voxel significance </th>\
    <th colspan=3> Coordinates in MNI referential</th>\
    <th>Cluster Size</th></tr>\n")
    page.write("<tr><th>p FWE corr<br>(Bonferroni)</th>\
    <th>p FDR corr</th><th>Z</th><th>p uncorr</th>")
    page.write("<th> x (mm) </th><th> y (mm) </th><th> z (mm) </th>\
    <th>(voxels)</th></tr>\n")

    for cluster in clusters:
        maxima = cluster['maxima']
        size = cluster['size']
        for j in range(min(len(maxima), nmaxima)):
            temp = ["%.3f" % cluster['fwer_p_value'][j]]
            temp.append("%.3f" % cluster['fdr_p_value'][j])
            temp.append("%.2f" % cluster['z_score'][j])
            temp.append("%.3f" % cluster['p_value'][j])
            for it in range(3):
                temp.append("%.0f" % maxima[j][it])
            if j == 0:
                # Main local maximum
                temp.append('%i' % size)
                page.write('<tr><th align="center">' + '</th>\
                <th align="center">'.join(temp) + '</th></tr>')
            else:
                # Secondary local maxima
                page.write('<tr><td align="center">' + '</td>\
                <td align="center">'.join(temp) + '</td><td></td></tr>\n')

    nclust = len(clusters)
    nvox = sum([clusters[k]['size'] for k in range(nclust)])

    # finish up
    page.write("</table>\n")
    page.write("Threshold Z: %.2f (%s control at %.3f)<br/>\n"
               % (z_threshold, method, p_threshold))
    page.write("Cluster level p-value threshold: %s<br/>\n" % cluster_pval)
    page.write("Cluster size threshold: %i voxels<br/>\n" % cluster_th)
    page.write("Number of voxels: %i<br>\n" % nvox)
    page.write("Number of clusters: %i<br>\n" % nclust)
    page.write("</center>\n")
    page.close()
    return clusters


def generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        z_maps,
        mask,
        design_matrices=None,
        subject_id=None,
        anat=None,
        display_mode="z",
        cut_coords=None,
        threshold=2.3,
        cluster_th=15,
        start_time=None,
        title=None,
        user_script_name=None,
        progress_logger=None,
        shutdown_all_reloaders=True,
        **glm_kwargs):
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

    design_matrix: list of 'DesignMatrix', `numpy.ndarray` objects or of
    strings (.png, .npz, etc.) for filenames
        design matrices for the experimental conditions

    contrasts: dict of arrays
       dictionary of contrasts of interest; the keys are the contrast ids,
       the values are contrast values (lists)

    z_maps: dict of 3D image objects or strings (image filenames)
       dict with same keys as 'contrasts'; the values are paths of z-maps
       for the respective contrasts

    anat: 3D array (optional)
        brain image to serve bg unto which activation maps will be plotted

    anat_affine: 2D array (optional)
        affine data for the anat

    threshold: float (optional)
        threshold to be applied to activation maps voxel-wise

    cluster_th: int (optional)
        minimal voxel count for clusteres declared as 'activated'

    cmap: cmap object (default viz.cm.cold_hot)
        color-map to use in plotting activation maps

    start_time: string (optional)
        start time for the stats analysis (useful for the generated
        report page)

    user_script_name: string (optional, default None)
        existing filename, path to user script used in doing the analysis

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
    # Delayed import of nipy for more robustness when it is not present
    from nipy.modalities.fmri.design_matrix import DesignMatrix

    # prepare for stats reporting
    if progress_logger is None:
        progress_logger = base_reporter.ProgressReport()

    output_dir = os.path.dirname(stats_report_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # copy css and js stuff to output dir
    base_reporter.copy_web_conf_files(output_dir)

    # initialize gallery of design matrices
    design_thumbs = base_reporter.ResultsGallery(
        loader_filename=os.path.join(output_dir,
                                     "design.html")
        )

    # initialize gallery of activation maps
    activation_thumbs = base_reporter.ResultsGallery(
        loader_filename=os.path.join(output_dir,
                                     "activation.html")
        )

    # get caller module handle from stack-frame
    if user_script_name is None:
        user_script_name = sys.argv[0]
    user_source_code = base_reporter.get_module_source_code(
        user_script_name)

    methods = """
    GLM and Statistical Inference have been done using the <i>%s</i> script, \
powered by <a href="%s">nipy</a>. Statistic images have been thresholded at \
Z>%s voxel-level.
    """ % (user_script_name, base_reporter.NIPY_URL, threshold)

    # report the control parameters used in the paradigm and analysis
    design_params = ""
    glm_kwargs["contrasts"] = contrasts
    if len(glm_kwargs):
        design_params += ("The following control parameters were used for  "
                    " specifying the experimental paradigm and fitting the "
                    "GLM:<br/><ul>")

        design_params += base_reporter.dict_to_html_ul(glm_kwargs)

    if start_time is None:
        start_time = base_reporter.pretty_time()

    if title is None:
        title = "GLM and Statistical Inference"
        if not subject_id is None:
            title += " for subject %s" % subject_id

    level1_html_markup = base_reporter.get_subject_report_stats_html_template(
        title=title,
        start_time=start_time,
        subject_id=subject_id,

        # insert source code stub
        source_script_name=user_script_name,
        source_code=user_source_code,

        design_params=design_params,
        methods=methods,
        threshold=threshold)

    with open(stats_report_filename, 'w') as fd:
        fd.write(str(level1_html_markup))
        fd.close()

    progress_logger.log("<b>Level 1 statistics</b><br/><br/>")

    # create design matrix thumbs
    if not design_matrices is None:
        if not hasattr(design_matrices, '__len__'):
            design_matrices = [design_matrices]

        for design_matrix, j in zip(design_matrices,
                                    xrange(len(design_matrices))):
            # sanitize design_matrix type
            if isinstance(design_matrix, basestring):
                if not isinstance(design_matrix, DesignMatrix):
                    if design_matrix.endswith('.npz'):
                        npz = np.load(design_matrix)
                        design_matrix = DesignMatrix(npz['X'],
                                                     npz['conditions'],
                                                     )
                else:
                    # XXX handle case of .png, jpeg design matrix image
                    raise TypeError(
                        "Unsupported design matrix type: %s" % type(
                            design_matrix))
            elif isinstance(design_matrix, np.ndarray) or isinstance(
                    design_matrix, list):
                X = np.array(design_matrix)
                conditions = ['%i' % i for i in xrange(X.shape[-1])]
                design_matrix = DesignMatrix(X, conditions)

            # plot design_matrix proper
            ax = design_matrix.show(rescale=True)
            ax.set_position([.05, .25, .9, .65])
            dmat_outfile = os.path.join(output_dir,
                                        'design_matrix_%i.png' % (j + 1),
                                        )
            pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)
            pl.close()

            thumb = base_reporter.Thumbnail()
            thumb.a = base_reporter.a(href=os.path.basename(dmat_outfile))
            thumb.img = base_reporter.img(src=os.path.basename(dmat_outfile),
                                     height="500px",
                                     )
            thumb.description = "Design Matrix"
            thumb.description += " %s" % (j + 1) if len(
                design_matrices) > 1 else ""

            # commit activation thumbnail into gallery
            design_thumbs.commit_thumbnails(thumb)

    # create activation thumbs
    for contrast_id in contrasts:
        z_map = z_maps[contrast_id]

        # load the map
        if isinstance(z_map, basestring):
            z_map = nibabel.load(z_map)

        # generate level 1 stats table
        title = "Level 1 stats for %s contrast" % contrast_id
        stats_table = os.path.join(output_dir, "%s_stats_table.html" % (
                contrast_id))
        generate_level1_stats_table(
            z_map, mask, stats_table, cluster_th=cluster_th,
            z_threshold=threshold, title=title)

        # plot activation proper
        # XXX: nilearn's plotting bug's about rotations inf affine, etc.
        z_map = reorder_img(z_map, resample="continuous")
        anat = reorder_img(anat, resample="continuous")
        plot_stat_map(z_map, anat, threshold=threshold,
                      display_mode=display_mode, cut_coords=cut_coords,
                      black_bg=True)

        # store activation plot
        z_map_plot = os.path.join(output_dir,
                                  "%s_z_map.png" % contrast_id)
        pl.savefig(z_map_plot, dpi=200, bbox_inches='tight', facecolor="k",
                   edgecolor="k")
        pl.close()

        # create thumbnail for activation
        thumbnail = base_reporter.Thumbnail()
        thumbnail.a = base_reporter.a(href=os.path.basename(stats_table))
        thumbnail.img = base_reporter.img(src=os.path.basename(z_map_plot),
                                          height="150px",)
        thumbnail.description = contrast_id
        activation_thumbs.commit_thumbnails(thumbnail)

    # we're done, shut down re-loaders
    progress_logger.log('<hr/>')

    # prevent stats report page from reloading henceforth
    progress_logger.finish(stats_report_filename)

    # prevent any related page from reloading
    if shutdown_all_reloaders:
        progress_logger.finish_dir(output_dir)

    # return generated html
    with open(stats_report_filename, 'r') as fd:
        stats_report = fd.read()
        fd.close()

        return stats_report


def group_one_sample_t_test(masks, effects_maps, contrasts, output_dir,
                            start_time=base_reporter.pretty_time(),
                            **kwargs):
    """
    Runs a one-sample t-test procedure for group analysis. Here, we are
    for each experimental condition, only interested refuting the null
    hypothesis H0: "The average effect accross the subjects is zero!"

    Parameters
    ----------
    masks: list of strings or nibabel image objects
        subject masks, one per subject

    effects_maps: list of dicts of lists
        effects maps from subject-level GLM; each entry is a dictionary;
        each entry (indexed by condition id) of this dictionary is the
        filename (or correspinding nibabel image object) for the effects
        maps for that condition (aka contrast),for that subject

    contrasts: dictionary of array_likes
        contrasts vectors, indexed by condition id

    kwargs: dict_like
        kwargs for plot_stats_map API
    """

    # make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert len(masks) == len(effects_maps), (len(masks), len(effects_maps))

    # compute group mask
    group_mask = nibabel.Nifti1Image(
        intersect_masks(masks).astype(np.int8),
        (nibabel.load(masks[0]) if isinstance(
                masks[0], basestring) else masks[0]).get_affine())

    # construct design matrix (only one covariate, namely the "mean effect")
    design_matrix = np.ones(len(effects_maps)
                            )[:, np.newaxis]  # only the intercept

    group_level_z_maps = {}
    group_level_t_maps = {}
    for contrast_id in contrasts:
        print "\tcontrast id: %s" % contrast_id

        # effects maps will be the input to the second level GLM
        first_level_image = nibabel.concat_images(
            [x[contrast_id] for x in effects_maps])

        # fit 2nd level GLM for given contrast
        group_model = FMRILinearModel(first_level_image,
                                      design_matrix, group_mask)
        group_model.fit(do_scaling=False, model='ols')

        # specify and estimate the contrast
        contrast_val = np.array(([[1.]])
                                )  # the only possible contrast !
        z_map, t_map = group_model.contrast(
            contrast_val, con_id='one_sample %s' % contrast_id, output_z=True,
            output_stat=True)

        # save map
        for map_type, map_img in zip(["z", "t"], [z_map, t_map]):
            map_dir = os.path.join(output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, 'group_level_%s.nii.gz' % (
                    contrast_id))
            print "\t\tWriting %s ..." % map_path
            nibabel.save(map_img, map_path)
            if map_type == "z":
                group_level_z_maps[contrast_id] = map_path
            elif map_type == "t":
                group_level_z_maps[contrast_id] = map_path

    # do stats report
    stats_report_filename = os.path.join(
        output_dir, "report_stats.html")
    generate_subject_stats_report(stats_report_filename, contrasts,
                                  group_level_z_maps, group_mask,
                                  start_time=start_time,
                                  **kwargs)

    print "\r\nStatistic report written to %s\r\n" % (
        stats_report_filename)

    return group_level_z_maps
