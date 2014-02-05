"""
:Module: nipy_glm_spm_auditory
Synopsis: Demo script for nipy's GLM and inference + reporting
on SPM's single-subject auditory data
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

import sys
import os
import pylab as pl
import numpy as np
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
import nibabel
import time
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData
                                                   )
import pypreprocess.reporting.glm_reporter as glm_reporter
import pypreprocess.reporting.base_reporter as base_reporter
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.io_utils import do_3Dto4D_merge

DATASET_DESCRIPTION = """\
<p>MoAEpilot <a href="http://www.fil.ion.ucl.ac.uk/spm/data/auditory/">\
SPM auditory dataset</a>.</p>
"""

if len(sys.argv)  < 3:
    print("\r\nUsage: python %s <spm_auditory_MoAEpilot_dir>"
          " <output_dir>\r\n") % sys.argv[0]
    print("Defaulting to current dir")
    # set data dir
    DATA_DIR = os.path.abspath('spm_auditory')

    # set output dir
    OUTPUT_DIR = os.path.abspath('spm_auditory_output')


else:
    # set data dir
    DATA_DIR = os.path.abspath(sys.argv[1])

    # set output dir
    OUTPUT_DIR = os.path.abspath(sys.argv[2])

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# construct experimental paradigm
stats_start_time = time.ctime()
tr = 7.
n_scans = 96
_duration = 6
epoch_duration = _duration * tr
conditions = ['rest', 'active'] * 8
duration = epoch_duration * np.ones(len(conditions))
onset = np.linspace(0, (len(conditions) - 1) * epoch_duration,
                    len(conditions))
paradigm = BlockParadigm(con_id=conditions, onset=onset, duration=duration)
hfcut = 2 * 2 * epoch_duration

# fetch spm auditory data
_subject_data = fetch_spm_auditory_data(DATA_DIR)
subject_data = SubjectData()
subject_data.func = _subject_data.func
subject_data.anat = _subject_data.anat
subject_data.subject_id = "sub001"
subject_data.output_dir = os.path.join(OUTPUT_DIR, subject_data.subject_id)

# preprocess the data
results = do_subjects_preproc(
    [subject_data],
    output_dir=OUTPUT_DIR,
    func_to_anat=True,
    # fwhm=8.,
    do_segment=False,
    do_normalize=False,
    dataset_id="SPM single-subject auditory",
    dataset_description=DATASET_DESCRIPTION,
    do_shutdown_reloaders=False,
    )

# collect preprocessed data
fmri_files = results[0]['func']
fmri_4D_filename = os.path.join(subject_data.output_dir,
                                "fmri_4D_preproc.nii.gz")
do_3Dto4D_merge(fmri_files, output_filename=fmri_4D_filename)
anat_file = results[0]['anat']

# construct design matrix
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
design_matrix = make_dmtx(frametimes,
                          paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

# plot and save design matrix
ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')
dmat_outfile = os.path.join(subject_data.output_dir, 'design_matrix.png')
pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)
pl.close()

# specify contrasts
contrasts = {}
n_columns = len(design_matrix.names)
for i in xrange(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

# more interesting contrasts"""
contrasts['active-rest'] = contrasts['active'] - contrasts['rest']

# fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FMRILinearModel(fmri_4D_filename,
                           design_matrix.matrix,
                           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

# save computed mask
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)

# compute bg unto which activation will be projected
anat_img = nibabel.load(anat_file)
anat = anat_img.get_data()
anat_affine = anat_img.get_affine()

print "Computing contrasts .."
z_maps = {}
for contrast_id, contrast_val in contrasts.iteritems():
    print "\tcontrast id: %s" % contrast_id
    z_map, t_map, eff_map, var_map = fmri_glm.contrast(
        contrasts[contrast_id],
        con_id=contrast_id,
        output_z=True,
        output_stat=True,
        output_effects=True,
        output_variance=True,
        )

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

        # collect zmaps for contrasts we're interested in
        if contrast_id == 'active-rest' and dtype == "z":
            z_maps[contrast_id] = map_path

        print "\t\t%s map: %s" % (dtype, map_path)

    print

# do stats report
stats_report_filename = os.path.join(subject_data.output_dir,
                                     "report_stats.html")
contrasts = dict((contrast_id, contrasts[contrast_id])
                 for contrast_id in z_maps.keys())
glm_reporter.generate_subject_stats_report(
    stats_report_filename,
    contrasts,
    z_maps,
    fmri_glm.mask,
    design_matrices=[design_matrix],
    subject_id=subject_data.subject_id,
    anat=anat,
    anat_affine=anat_affine,
    slicer='z',
    cut_coords=5,
    cluster_th=50,  # we're only interested in this 'large' clusters
    start_time=stats_start_time,

    # additional ``kwargs`` for more informative report
    paradigm=paradigm.__dict__,
    TR=tr,
    n_scans=n_scans,
    hfcut=hfcut,
    frametimes=frametimes,
    drift_model=drift_model,
    hrf_model=hrf_model,
    )

base_reporter.ProgressReport().finish_dir(OUTPUT_DIR)

print "\r\nStatistic report written to %s\r\n" % stats_report_filename
