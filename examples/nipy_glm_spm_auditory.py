"""
Synopsis: Demo script for nipy's GLM tools
Author: dohmatob elvis dopgima

"""

import sys
import os
import pylab as pl
import numpy as np
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs import viz
import nibabel
import time

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

import nipype_preproc_spm_utils
import reporting.reporter as reporter
from datasets_extras import fetch_spm_auditory_data
from io_utils import do_3Dto4D_merge

DATASET_DESCRIPTION = """\
<p>MoAEpilot <a href="http://www.fil.ion.ucl.ac.uk/spm/data/auditory/">\
SPM auditory dataset</a>.</p>\
"""

"""MISC"""
NIPY_URL = "http://nipy.sourceforge.net/nipy/stable/index.html"

if len(sys.argv)  < 3:
    print ("\r\nUsage: python %s <spm_auditory_MoAEpilot_dir>"
           " <output_dir>\r\n") % sys.argv[0]
    sys.exit(1)

"""set data dir"""
DATA_DIR = os.path.abspath(sys.argv[1])

# set output dir
OUTPUT_DIR = os.path.abspath(sys.argv[2])
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


"""fetch spm auditory data"""
_subject_data = fetch_spm_auditory_data(DATA_DIR)

subject_data = nipype_preproc_spm_utils.SubjectData()
subject_data.func = _subject_data["func"]
subject_data.anat = _subject_data["anat"]
subject_data.output_dir = os.path.join(
    OUTPUT_DIR, subject_data.subject_id)

"""preprocess the data"""
report_filename = os.path.join(OUTPUT_DIR,
                               "_report.html")
results = nipype_preproc_spm_utils.do_subjects_preproc(
    [subject_data],
    fwhm=[6, 6, 6],
    dataset_description=DATASET_DESCRIPTION,
    report_filename=report_filename,
    )

"""prepare for stats reporting"""
progress_logger = results[0]['progress_logger']
stats_report_filename = os.path.join(subject_data.output_dir,
                                     "report_stats.html")
progress_logger.watch_file(stats_report_filename)
level1_loader_filename = os.path.join(
    subject_data.output_dir, "level1_thumbs.html")
level1_thumbs = reporter.ResultsGallery(
    loader_filename=level1_loader_filename,
    )
level1_html_markup = reporter.FSL_SUBJECT_REPORT_STATS_HTML_TEMPLATE(
    ).substitute(
    results=level1_thumbs,
    start_time=time.ctime(),
    subject_id=subject_data.subject_id,
    methods=('GLM and inference have been done using '
             '<a href="%s">nipy</a>.') % NIPY_URL,
             )
with open(stats_report_filename, 'w') as fd:
    fd.write(str(level1_html_markup))
    fd.close()
progress_logger.log("<b>Level 1 statistics</b><br/><br/>")

"""collect preprocessed data"""
fmri_data = results[0]['func']
fmri_data = do_3Dto4D_merge(fmri_data)

"""construct experimental paradigm"""
progress_logger.log("Computing design matrix ..<br/>")
tr = 7
n_scans = 96
_duration = 6
epoch_duration = _duration * tr
conditions = ['rest', 'active'] * 8
duration = epoch_duration * np.ones(len(conditions))
onset = np.linspace(0, (len(conditions) - 1) * epoch_duration,
                    len(conditions))
paradigm = BlockParadigm(con_id=conditions, onset=onset, duration=duration)
hfcut = 2 * 2 * epoch_duration

"""construct design matrix"""
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
design_matrix = make_dmtx(frametimes,
                          paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

"""show design matrix"""
ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')
dmat_outfile = os.path.join(subject_data.output_dir, 'design_matrix.png')
pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)
thumb = reporter.Thumbnail()
thumb.a = reporter.a(href=os.path.basename(dmat_outfile))
thumb.img = reporter.img(src=os.path.basename(dmat_outfile),
                         height="500px",
                         width="600px")
thumb.description = "Design Matrix"
level1_thumbs.commit_thumbnails(thumb)

"""specify contrasts"""
contrasts = {}
n_columns = len(design_matrix.names)
for i in xrange(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

"""more interesting contrasts"""
contrasts['active-rest'] = contrasts['active'] - contrasts['rest']

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

"""level 1 (within subject) inference"""
print "Computing contrasts .."
progress_logger.log("Computing contrasts ..<br/>")
for contrast_id in contrasts:
    print "\tcontrast id: %s" % contrast_id
    z_map, t_map, eff_map, var_map = fmri_glm.contrast(
        contrasts[contrast_id],
        con_id=contrast_id,
        output_z=True,
        output_stat=True,
        output_effects=True,
        output_variance=True,)

    if contrast_id == 'active-rest':
        z_threshold = 5.
        z_axis_max = np.unravel_index(
            z_map.get_data().argmax(), z_map.shape)[2]
        z_axis_min = np.unravel_index(
            z_map.get_data().argmin(), z_map.shape)[2]
        pos_data = z_map.get_data() * (z_map.get_data() > 0)
        vmax = pos_data.max()
        vmin = - vmax

        viz.plot_map(pos_data, z_map.get_affine(),
                     cmap=pl.cm.hot,
                     vmin=vmin,
                     vmax=vmax,
                     threshold=z_threshold,
                     slicer='z',
                     cut_coords=range(z_axis_min, z_axis_max + 1, 2),
                     black_bg=True,
                     )

        z_map_plot = os.path.join(subject_data.output_dir,
                                  "activation_map.png")
        stats_table = os.path.join(subject_data.output_dir, "stats_table.html")
        pl.savefig(z_map_plot, dpi=200, bbox_inches='tight',
                   facecolor="k",
                   edgecolor="k")

        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(href=os.path.basename(z_map_plot))
        thumbnail.img = reporter.img(
            src=os.path.basename(z_map_plot), height="250px")
        thumbnail.description = "Activation z-map (thresholded at %s)"\
            % z_threshold
        level1_thumbs.commit_thumbnails(thumbnail)

    for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                              [z_map, t_map, eff_map, var_map]):
        map_dir = os.path.join(
            subject_data.output_dir, '%s_maps' % dtype)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(
            map_dir, '%s.nii.gz' % contrast_id)
        nibabel.save(out_map, map_path)

        if contrast_id == "active-rest" and dtype == "z":
            reporter.generate_level1_report(
                z_map, fmri_glm.mask,
                stats_table,
                title="%s z-map: %s" % (contrast_id, map_path),
                cluster_th=50)

        print "\t\t%s map: %s" % (dtype, map_path)

    print

"""We're done"""
progress_logger.log('<hr/>')
progress_logger.finish_all()

"""show all generated plots this far"""
# pl.show()
