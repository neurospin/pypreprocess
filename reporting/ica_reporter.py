"""
:Module: ica_reporter
:Synopsis: Report generation after ICA
:Author: dohmatob elvis dopgima

"""

import os
import sys
import pylab as pl
from nipy.labs import viz
import nibabel
import numpy as np
import base_reporter
import time


def generate_ica_report(
    stats_report_filename,
    ica_maps,
    mask=None,
    report_title='ICA Report',
    methods='ICA',
    anat=None,
    anat_affine=None,
    threshold=2.,
    cluster_th=0,
    cmap=viz.cm.cold_hot,
    start_time=None,
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

    # prepare for stats reporting
    if progress_logger is None:
        progress_logger = base_reporter.ProgressReport()

    output_dir = os.path.dirname(stats_report_filename)

    # copy css and js stuff to output dir
    base_reporter.copy_web_conf_files(output_dir)

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

    if start_time is None:
        start_time = time.ctime()

    ica_html_markup = base_reporter.get_ica_html_template(
        ).substitute(
        title=report_title,
        start_time=start_time,

        # insert source code stub
        source_script_name=user_script_name,
        source_code=user_source_code,

        methods=methods,
        cmap=cmap.name)

    with open(stats_report_filename, 'w') as fd:
        fd.write(str(ica_html_markup))
        fd.close()

    progress_logger.log("<b>ICA</b><br/><br/>")

    # make colorbar (place-holder, will be overridden, once we've figured out
    # the correct end points) for activations
    colorbar_outfile = os.path.join(output_dir,
                                    'activation_colorbar.png')
    base_reporter.make_standalone_colorbar(
        cmap, threshold, 8., colorbar_outfile)

    # create activation thumbs
    _vmax = 0
    _vmin = threshold
    for ica_map_id, ica_map in ica_maps.iteritems():
        # load the map
        if isinstance(ica_map, basestring):
            ica_map = nibabel.load(ica_map)

        # compute cut_coords for viz.plot_map(..) API
        cut_coords = base_reporter.get_cut_coords(
            ica_map.get_data(), n_axials=12, delta_z_axis=3)

        # compute vmin and vmax
        vmin, vmax = base_reporter.compute_vmin_vmax(ica_map.get_data())

        # update colorbar endpoints
        _vmax = max(_vmax, vmax)
        _vmin = min(_vmin, vmin)

        # plot activation proper
        viz.plot_map(ica_map.get_data(), ica_map.get_affine(),
                     cmap=cmap,
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
        ica_map_plot = os.path.join(output_dir,
                                  "%s_ica_map.png" % ica_map_id)
        pl.savefig(ica_map_plot, dpi=200, bbox_inches='tight',
                   facecolor="k",
                   edgecolor="k")
        stats_table = ica_map_plot  # os.path.join(output_dir,
                                   # "%s_stats_table.html" % ica_map_id)

        # create thumbnail for activation
        thumbnail = base_reporter.Thumbnail()
        thumbnail.a = base_reporter.a(href=os.path.basename(stats_table))
        thumbnail.img = base_reporter.img(
            src=os.path.basename(ica_map_plot), height="200px",)
        thumbnail.description = "Component: %s" % ica_map_id
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
