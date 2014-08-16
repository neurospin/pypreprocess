"""
:Module: nipype_preproc_spm_auditory
Synopsis: Minimal script for preprocessing single-subject data
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

import sys
import os
import time
import numpy as np
import pylab as pl
import nibabel
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report

# file containing configuration for preprocessing the data
this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
jobfile = os.path.join(this_dir, "spm_auditory_preproc.ini")

# set dataset dir
if len(sys.argv) > 1:
    dataset_dir = sys.argv[1]
else:
    dataset_dir = os.path.join(this_dir, "spm_auditory")

# fetch spm auditory data
sd = fetch_spm_auditory_data(dataset_dir)

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
fd = open(sd.func[0].split(".")[0] + "_onset.txt", "w")
for c, o, d in zip(conditions, onset, duration):
    fd.write("%s %s %s\r\n" % (c, o, d))
fd.close()
assert 0

# preprocess the data
subject_data = do_subjects_preproc(jobfile, dataset_dir=dataset_dir)[0]

# construct design matrix
nscans = len(subject_data.func[0])
frametimes = np.linspace(0, (nscans - 1) * tr, nscans)
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

# specify contrasts
contrasts = {}
n_columns = len(design_matrix.names)
for i in xrange(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

# more interesting contrasts"""
contrasts['active-rest'] = contrasts['active'] - contrasts['rest']

# fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FMRILinearModel(nibabel.concat_images(subject_data.func[0]),
                           design_matrix.matrix,
                           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

# save computed mask
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)

# compute bg unto which activation will be projected
anat_img = nibabel.load(subject_data.anat)
anat = anat_img.get_data()
anat_affine = anat_img.get_affine()

print "Computing contrasts .."
z_maps = {}
effects_maps = {}
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
stats_report_filename = os.path.join(subject_data.reports_output_dir,
                                     "report_stats.html")
contrasts = dict((contrast_id, contrasts[contrast_id])
                 for contrast_id in z_maps.keys())
generate_subject_stats_report(
    stats_report_filename,
    contrasts,
    z_maps,
    fmri_glm.mask,
    design_matrices=[design_matrix],
    subject_id=subject_data.subject_id,
    anat=anat,
    anat_affine=anat_affine,
    slicer='ortho',
    threshold=3.,
    cluster_th=50,  # we're only interested in this 'large' clusters
    start_time=stats_start_time,

    # additional ``kwargs`` for more informative report
    paradigm=paradigm.__dict__,
    TR=tr,
    nscans=nscans,
    hfcut=hfcut,
    frametimes=frametimes,
    drift_model=drift_model,
    hrf_model=hrf_model,
    )

print "\r\nStatistic report written to %s\r\n" % stats_report_filename
