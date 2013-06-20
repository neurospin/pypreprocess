"""
:Module: spm_multimodal_fmri
:Synopsis: script for preproc + stats on SPM multi-modal face data set
http://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
import nibabel as ni
import scipy.io
import time
import glob
import sys
import os

warning = ("%s: THIS SCRIPT MUST BE RUN FROM ITS PARENT "
           "DIRECTORY!") % sys.argv[0]
banner = "#" * len(warning)
separator = "\r\n\t"

print separator.join(['', banner, warning, banner, ''])

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])
sys.path.append(PYPREPROCESS_DIR)

# import pypreprocess plugins
import nipype_preproc_spm_utils
import reporting.glm_reporter as glm_reporter
from external.nisl.datasets import fetch_spm_multimodal_fmri_data

# set data and output paths (change as you will)
DATA_DIR = "spm_multimodal_fmri"
print "\tDATA_DIR: %s" % DATA_DIR
OUTPUT_DIR = "spm_multimodal_runs"
print "\tOUTPUT_DIR: %s" % OUTPUT_DIR
print
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DO_REPORT = True

# fetch the data
_subject_data = fetch_spm_multimodal_fmri_data(DATA_DIR)
subject_data = nipype_preproc_spm_utils.SubjectData()

subject_data.subject_id = "sub001"
subject_data.session_id = ["Session1", "Session2"]
subject_data.func = [_subject_data.func1, _subject_data.func2]
subject_data.anat = _subject_data.anat

subject_data.output_dir = os.path.join(OUTPUT_DIR,
                                       subject_data.subject_id)


# preprocess the data
results = nipype_preproc_spm_utils.do_subjects_preproc(
    [subject_data],
    output_dir=OUTPUT_DIR,
    fwhm=[8, 8, 8],
    dataset_id="SPM MULTIMODAL (see @alex)",
    do_shutdown_reloaders=False,
    )

# collect preprocessed data
fmri_imgs = [ni.concat_images(session_func)
             for session_func in results[0]['func']]
anat_img = ni.load(results[0]['anat'])

# experimental paradigm meta-params
stats_start_time = time.ctime()
tr = 2.
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
hfcut = 128.

# make design matrices
design_matrices = []
for x in xrange(2):
    n_scans = fmri_imgs[x].shape[-1]

    timing = scipy.io.loadmat(os.path.join(DATA_DIR,
                                           "fMRI/trials_ses%i.mat" % (x + 1)),
                              squeeze_me=True, struct_as_record=False)

    faces_onsets = timing['onsets'][0].ravel()
    scrambled_onsets = timing['onsets'][1].ravel()
    onsets = np.hstack((faces_onsets, scrambled_onsets))
    onsets *= tr  # because onsets were reporting in 'scans' units
    conditions = ['faces'] * len(faces_onsets) + ['scrambled'] * len(
        scrambled_onsets)
    paradigm = EventRelatedParadigm(conditions, onsets)
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    design_matrix = make_dmtx(frametimes,
                              paradigm, hrf_model=hrf_model,
                              drift_model=drift_model, hfcut=hfcut)

    design_matrices.append(design_matrix)

# specify contrasts
contrasts = {}
n_columns = len(design_matrix.names)
for i in xrange(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

# more interesting contrasts
contrasts['faces-scrambled'] = contrasts['faces'] - contrasts['scrambled']
contrasts['scrambled-faces'] = contrasts['scrambled'] - contrasts['faces']
contrasts['effects_of_interest'] = contrasts['faces'] + contrasts['scrambled']

# we've thesame contrasts over sessions, so let's replicate
contrasts = dict((contrast_id, [contrast_val] * 2)
                 for contrast_id, contrast_val in contrasts.iteritems())

# fit GLM
print('\r\nFitting a GLM (this takes time)...')
fmri_glm = FMRILinearModel(fmri_imgs,
                           [dmat.matrix for dmat in design_matrices],
                           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

# save computed mask
mask_path = os.path.join(OUTPUT_DIR, "mask.nii.gz")
print "Saving mask image %s" % mask_path
ni.save(fmri_glm.mask, mask_path)

print "Computing contrasts .."
z_maps = {}
for contrast_id, contrast_val in contrasts.iteritems():
    print "\tcontrast id: %s" % contrast_id
    z_map, t_map, eff_map, var_map = fmri_glm.contrast(
        contrast_val,
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
            OUTPUT_DIR, '%s_maps' % dtype)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(
            map_dir, '%s.nii.gz' % contrast_id)
        ni.save(out_map, map_path)

        # collect zmaps for contrasts we're interested in
        if dtype == 'z':
            z_maps[contrast_id] = map_path

        print "\t\t%s map: %s" % (dtype, map_path)

# do stats report
if DO_REPORT:
    stats_report_filename = os.path.join(subject_data.output_dir,
                                         "report_stats.html")
    contrasts = dict((contrast_id, contrasts[contrast_id])
                     for contrast_id in z_maps.keys())
    glm_reporter.generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        z_maps,
        fmri_glm.mask,
        anat=anat_img.get_data(),
        anat_affine=anat_img.get_affine(),
        design_matrices=design_matrices,
        subject_id="sub001",
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

    print "\r\nStatistic report written to %s\r\n" % stats_report_filename
