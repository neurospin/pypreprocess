import os
import sys
import pylab as pl
from ..external.nipy_labs import viz
import nibabel
import numpy as np
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs.mask import intersect_masks
import base_reporter


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
    # Delayed import of nipy for more robustness when it is not present
    import nipy.labs.statistical_mapping as sm

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

    # do some sanity checks
    if title is None:
        title = "Level 1 Statistics"

    # clusters, info = sm.cluster_stats(zmap, mask, height_th=p_threshold,
    #                                   nulls=nulls, cluster_th=cluster_th,)

    clusters, _ = sm.cluster_stats(zmap, mask, height_th=p_threshold,
                                      nulls=nulls, cluster_th=cluster_th,)

    if clusters is not None:
        clusters = [c for c in clusters if c['cluster_pvalue'] < cluster_pval]

    #if clusters == None or info == None:
    #    print "No results were written for %s" % zmap_file_path
    #    return
    if clusters == None:
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
            temp = ["%.3f" % cluster['fwer_pvalue'][j]]
            temp.append("%.3f" % cluster['fdr_pvalue'][j])
            temp.append("%.2f" % cluster['zscore'][j])
            temp.append("%.3f" % cluster['pvalue'][j])
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

    page.write("</table>\n")
    page.write("Threshold Z: %.2f (%s control at %.3f)<br/>\n" \
                   % (z_threshold, method, p_threshold))
    page.write("Cluster level p-value threshold: %s<br/>\n" % cluster_pval)
    page.write("Cluster size threshold: %i voxels<br/>\n" % cluster_th)
    page.write("Number of voxels: %i<br>\n" % nvox)
    page.write("Number of clusters: %i<br>\n" % nclust)

    # finish up
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
    anat_affine=None,
    slicer="z",
    cut_coords=None,
    statistical_mapping_trick=False,
    threshold=2.3,
    cluster_th=0,
    cmap=viz.cm.cold_hot,
    start_time=None,
    title=None,
    user_script_name=None,
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
        brain image to serve bg unto which activation maps will be plotted;
        passed to viz.plot_map API

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

    if slicer == "ortho":
        statistical_mapping_trick = False

        if not hasattr(cut_coords, "__iter__"):
            cut_coords = None
    
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
        threshold=threshold,
        cmap=cmap.name)

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
                design_matrix,
                list):
                X = np.array(design_matrix)
                # assert len(X.shape) == 2
                conditions = ['%i' % i for i in xrange(X.shape[-1])]
                design_matrix = DesignMatrix(X, conditions)
            # else:
            #     raise TypeError(
            #         "Unsupported design matrix type '%s'" % type(
            #             design_matrix))

            # plot design_matrix proper
            ax = design_matrix.show(rescale=True)
            ax.set_position([.05, .25, .9, .65])
            dmat_outfile = os.path.join(output_dir,
                                        'design_matrix_%i.png' % (j + 1),
                                        )
            pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)

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

    # make colorbar (place-holder, will be overridden, once we've figured out
    # the correct end points) for activations
    colorbar_outfile = os.path.join(output_dir,
                                    'activation_colorbar.png')
    base_reporter.make_standalone_colorbar(
        cmap, threshold, 8., colorbar_outfile)

    # create activation thumbs
    _vmax = 0
    _vmin = threshold
    for contrast_id, contrast_val in contrasts.iteritems():
        z_map = z_maps[contrast_id]

        # load the map
        if isinstance(z_map, basestring):
            z_map = nibabel.load(z_map)

        # compute vmin and vmax
        vmin, vmax = base_reporter.compute_vmin_vmax(z_map.get_data())

        # update colorbar endpoints
        _vmax = max(_vmax, vmax)
        _vmin = min(_vmin, vmin)

        # sanitize anat
        if not anat is None:
            if anat.ndim == 4:
                assert anat.shape[-1] == 1, (
                    "Expecting 3D array for ant, got shape %s" % str(
                        anat.shape))
                anat = anat[..., 0]
            else:
                assert anat.ndim == 3, (
                    "Expecting 3D array for ant, got shape %s" % str(
                        anat.shape))

        # generate level 1 stats table
        title = "Level 1 stats for %s contrast" % contrast_id
        stats_table = os.path.join(output_dir, "%s_stats_table.html" % (
                contrast_id))
        z_clusters = generate_level1_stats_table(
            z_map, mask,
            stats_table,
            cluster_th=cluster_th,
            z_threshold=threshold,
            title=title,
            )

        # compute cut_coords
        if statistical_mapping_trick:
            slicer = slicer.lower()
            if slicer in ["x", "y", "z"] and cut_coords is None or isinstance(
                cut_coords, int):
                axis = 'xyz'.index(slicer)
                if cut_coords is None:
                    cut_coords = min(5, len(z_clusters))
                _cut_coords = [x for zc in z_clusters for x in list(
                        set(zc['maxima'][..., axis]))]
                # # _cut_coords = _maximally_separated_subset(
                # #     _cut_coords, cut_coords)

                # assert len(_cut_coords) == cut_coords
                cut_coords = _cut_coords

        # plot activation proper
        viz.plot_map(z_map.get_data(), z_map.get_affine(),
                     cmap=cmap,
                     anat=anat,
                     anat_affine=anat_affine,
                     threshold=threshold,
                     slicer=slicer,
                     cut_coords=cut_coords,
                     black_bg=True
                     )

        # store activation plot
        z_map_plot = os.path.join(output_dir,
                                  "%s_z_map.png" % contrast_id)
        pl.savefig(z_map_plot, dpi=200, bbox_inches='tight', facecolor="k",
                   edgecolor="k")

        # create thumbnail for activation
        thumbnail = base_reporter.Thumbnail()
        thumbnail.a = base_reporter.a(href=os.path.basename(stats_table))
        thumbnail.img = base_reporter.img(src=os.path.basename(z_map_plot),
                                          height="150px",)
        thumbnail.description = contrast_id
        activation_thumbs.commit_thumbnails(thumbnail)

    # make colorbar for activations
    base_reporter.make_standalone_colorbar(
        cmap, _vmin, _vmax, colorbar_outfile)

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
        parameters can be regular `nipy.labs.viz.plot_map` parameters
        (e.g slicer="y") or any parameter we want be reported (e.g
        fwhm=[5, 5, 5])

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
        z_map, = group_model.contrast(contrast_val,
                                      con_id='one_sample %s' % contrast_id,
                                      output_z=True)

        # save map
        map_dir = os.path.join(output_dir, 'z_maps')
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(map_dir, 'group_level_%s.nii.gz' % (
                contrast_id))
        print "\t\tWriting %s ..." % map_path
        nibabel.save(z_map, map_path)
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
