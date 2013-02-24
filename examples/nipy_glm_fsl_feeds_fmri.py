"""
Synopsis: nipy demo script
Author: dohmatob elvis dopgima

"""

import os
import sys
import pylab as pl
import matplotlib as mpl
import numpy as np
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs import viz
import nibabel
import time

"""path trick"""
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

"""import utilities for preproc, reporting, and io"""
import nipype_preproc_spm_utils
import reporting.reporter as reporter
from datasets_extras import unzip_nii_gz
from io_utils import compute_mean_3D_image

"""MISC"""
NIPY_URL = "http://nipy.sourceforge.net/nipy/stable/index.html"
z_threshold = 2.3

data_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
unzip_nii_gz(data_dir)

"""fetch input data"""
subject_data = nipype_preproc_spm_utils.SubjectData()
subject_data.subject_id = "sub001"
subject_data.anat = os.path.join(data_dir, "structural_brain.nii")
subject_data.func = os.path.join(data_dir, "fmri.nii")
subject_data.output_dir = os.path.join(
    output_dir, subject_data.subject_id)

"""preprocess the data"""
report_filename = os.path.join(output_dir,
                               "_report.html")
results = nipype_preproc_spm_utils.do_subjects_preproc(
    [subject_data],
    output_dir=output_dir,
    fwhm=[5, 5, 5],
    report_filename=report_filename,
    )

"""collect preprocessed data"""
fmri_data = results[0]['func']

"""prepare for stats reporting"""
progress_logger = results[0]['progress_logger']
stats_report_filename = os.path.join(subject_data.output_dir,
                                     "report_stats.html")
progress_logger.watch_file(stats_report_filename)
design_thumbs = reporter.ResultsGallery(
    loader_filename=os.path.join(subject_data.output_dir,
                                 "design.html")
    )
activation_thumbs = reporter.ResultsGallery(
    loader_filename=os.path.join(subject_data.output_dir,
                                 "activation.html")
    )

methods = """
GLM and inference have been done using <a href="%s">nipy</a>. Statistic \
images have been thresholded at Z>%s voxel-level.
""" % (NIPY_URL, z_threshold)

level1_html_markup = reporter.FSL_SUBJECT_REPORT_STATS_HTML_TEMPLATE(
    ).substitute(
    start_time=time.ctime(),
    subject_id=subject_data.subject_id,
    methods=methods)
with open(stats_report_filename, 'w') as fd:
    fd.write(str(level1_html_markup))
    fd.close()
progress_logger.log("<b>Level 1 statistics</b><br/><br/>")

"""experimental setup"""
n_scans = 180
TR = 3.0
TR * n_scans
TR * n_scans / 60
TR * n_scans / 90
np.linspace(0, TR * (n_scans - 1), 9)
EV1_epoch_duration = 2 * 30
EV2_epoch_duration = 2 * 45
TA = TR * n_scans
EV1_epochs = TA / EV1_epoch_duration
EV1_epochs = int(TA / EV1_epoch_duration)
EV2_epochs = int(TA / EV2_epoch_duration)
EV1_onset = np.linspace(0, EV1_epoch_duration * (EV1_epochs - 1), EV1_epochs)
EV2_onset = np.linspace(0, EV2_epoch_duration * (EV2_epochs - 1), EV2_epochs)
EV1_on = 30
EV2_on = 45
conditions = ['EV1'] * EV1_epochs + ['EV2'] * EV2_epochs
onset = list(EV1_onset) + list(EV2_onset)
duration = [EV1_on] * EV1_epochs + [EV2_on] * EV2_epochs
paradigm = BlockParadigm(con_id=conditions, onset=onset, duration=duration)
frametimes = np.linspace(0, (n_scans - 1) * TR, n_scans)
hfcut = 135

"""construct design matrix"""
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
design_matrix = make_dmtx(frametimes,
                          paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

"""show design matrix"""
ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
dmat_outfile = os.path.join(subject_data.output_dir, 'design_matrix.png')
pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)
thumb = reporter.Thumbnail()
thumb.a = reporter.a(href=os.path.basename(dmat_outfile))
thumb.img = reporter.img(src=os.path.basename(dmat_outfile),
                         height="400px",
                         )
thumb.description = "Design Matrix"
design_thumbs.commit_thumbnails(thumb)

"""specify contrasts"""
contrasts = {}
n_columns = len(design_matrix.names)
I = np.eye(len(design_matrix.names))
for i in xrange(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = I[2 * i]

"""more interesting contrasts"""
contrasts['EV1>EV2'] = contrasts['EV1'] - contrasts['EV2']
contrasts['EV2>EV1'] = contrasts['EV2'] - contrasts['EV1']

"""fit GLM"""
print('\r\nFitting a GLM (this takes time) ..')
progress_logger.log("Fitting GLM ..<br/>")
fmri_glm = FMRILinearModel(fmri_data, design_matrix.matrix,
           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

"""save computed mask"""
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)


def make_standalone_colorbar(vmin, vmax, colorbar_outfile=None):
    """Plots a stand-alone colorbar

    """

    fig = pl.figure(figsize=(8, 1.5))
    ax = fig.add_axes([0.05, 0.4, 0.9, 0.5])

    cmap = pl.cm.hot
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
    pl.savefig(colorbar_outfile)

    return cb


"""level 1 (within subject) inference"""
_vmax = 0
_vmin = z_threshold
print "Computing contrasts .."
progress_logger.log("Computing contrasts ..<br/>")
for contrast_id, contrast_val in contrasts.iteritems():
    print "\tcontrast id: %s" % contrast_id
    z_map, t_map, eff_map, var_map = fmri_glm.contrast(
        contrasts[contrast_id],
        con_id=contrast_id,
        output_z=True,
        output_stat=True,
        output_effects=True,
        output_variance=True,)

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

    # compute bg unto which activation will be projected
    mean_fmri_data = compute_mean_3D_image(fmri_data)
    anat = mean_fmri_data.get_data()
    anat_affine = mean_fmri_data.get_affine()

    # plot activation proper
    viz.plot_map(pos_data, z_map.get_affine(),
                 cmap=pl.cm.hot,
                 anat=anat,
                 anat_affine=anat_affine,
                 vmin=vmin,
                 vmax=vmax,
                 threshold=z_threshold,
                 slicer='z',
                 cut_coords=cut_coords,
                 black_bg=True,
                 )

    # store activation plot
    z_map_plot = os.path.join(subject_data.output_dir,
                              "%s_z_map.png" % contrast_id)
    pl.savefig(z_map_plot, dpi=200, bbox_inches='tight',
               facecolor="k",
               edgecolor="k")

    # create thumbnail for activation
    thumbnail = reporter.Thumbnail()
    thumbnail.a = reporter.a(href=os.path.basename(z_map_plot))
    thumbnail.img = reporter.img(
        src=os.path.basename(z_map_plot), height="250px",)
    thumbnail.description = "%s contrast: %s" % (contrast_id, contrast_val)
    activation_thumbs.commit_thumbnails(thumbnail)

    # store stat maps to disk
    for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                              [z_map, t_map, eff_map, var_map]):
        map_dir = os.path.join(
            subject_data.output_dir, '%s_maps' % dtype)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(
            map_dir, '%s.nii.gz' % contrast_id)
        nibabel.save(out_map, map_path)

        if contrast_id == "EV1" and dtype == "z":
            stats_table = os.path.join(subject_data.output_dir,
                                       "stats_table.html")
            reporter.generate_level1_report(
                z_map, fmri_glm.mask,
                stats_table,
                title="%s z-map: %s" % (contrast_id, map_path),
                cluster_th=15,  # not interested in clusters with < 15 voxels
                )

        print "\t\t%s map: %s" % (dtype, map_path)

    print

"""make colorbar for activations"""
colorbar_outfile = os.path.join(subject_data.output_dir,
                                'activation_colorbar.png')
make_standalone_colorbar(_vmin, _vmax, colorbar_outfile)

"""We're done"""
progress_logger.log('<hr/>')
progress_logger.finish_all()

"""show all generated plots this far"""
# pl.show()
