"""
:Module: nipy_glm_fsl_feeds_fmri
Synopsis: Demo script for nipy's GLM and inference + reporting
on FSL's FEEDS fMRI single-subject example data
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

import os
import sys
import numpy as np
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
import nibabel
import time

"""path trick"""
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))

"""import utilities for preproc, reporting, and io"""
import nipype_preproc_spm_utils
import reporting.glm_reporter as glm_reporter
import reporting.base_reporter as base_reporter
from external.nilearn.datasets import fetch_fsl_feeds_data
from datasets_extras import unzip_nii_gz
from io_utils import compute_mean_3D_image

"""MISC"""
DATASET_DESCRIPTION = "FSL FEADS example data (single-subject)"

"""sanitize cmd line"""
if len(sys.argv)  < 3:
    print ("\r\nUsage: python %s <path to FSL feeds data directory>"
           " <output_dir>\r\n") % sys.argv[0]
    print ("Example:\r\npython %s /usr/share/fsl-feeds/data/"
           " fsl_feeds_fmri_runs") % sys.argv[0]
    sys.exit(1)

"""set data dir"""
data_dir = os.path.abspath(sys.argv[1])

"""set output dir"""
output_dir = os.path.abspath(sys.argv[2])
unzip_nii_gz(data_dir)

"""experimental setup"""
stats_start_time = time.ctime()
n_scans = 180
TR = 3.
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
maximum_epoch_duration = max(EV1_epoch_duration, EV2_epoch_duration)
hfcut = 1.5 * maximum_epoch_duration  # why ?

"""construct design matrix"""
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
design_matrix = make_dmtx(frametimes,
                          paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

"""fetch input data"""
_subject_data = fetch_fsl_feeds_data(data_dir)
subject_data = nipype_preproc_spm_utils.SubjectData()
subject_data.subject_id = "sub001"
subject_data.func = _subject_data.func.rstrip('.gz')
unzip_nii_gz(os.path.dirname(subject_data.func))
subject_data.anat = _subject_data.anat.rstrip('.gz')
subject_data.output_dir = os.path.join(
    output_dir, subject_data.subject_id)
unzip_nii_gz(os.path.dirname(subject_data.anat))

"""preprocess the data"""
results = nipype_preproc_spm_utils.do_subjects_preproc(
    [subject_data],
    output_dir=output_dir,
    # fwhm=[5, 5, 5],
    dataset_id="FSL FEEDS single-subject",
    dataset_description=DATASET_DESCRIPTION,
    do_shutdown_reloaders=False,
    )

"""collect preprocessed data"""
fmri_files = results[0]['func']
anat_file = results[0]['anat']

"""specify contrasts"""
contrasts = {}
n_columns = len(design_matrix.names)
I = np.eye(len(design_matrix.names))
for i in xrange(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = I[2 * i]

"""more interesting contrasts"""
contrasts['EV1>EV2'] = contrasts['EV1'] - contrasts['EV2']
contrasts['EV2>EV1'] = contrasts['EV2'] - contrasts['EV1']
contrasts['effects_of_interest'] = contrasts['EV1'] + contrasts['EV2']

"""fit GLM"""
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FMRILinearModel(fmri_files, design_matrix.matrix,
           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

"""save computed mask"""
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)

# compute bg unto which activation will be projected
mean_fmri_files = compute_mean_3D_image(fmri_files)
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

        if dtype == "z":
            z_maps[contrast_id] = map_path

        print "\t\t%s map: %s" % (dtype, map_path)

    print

"""do stats report"""
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
    cluster_th=50,  # we're only interested in this 'large' clusters
    start_time=stats_start_time,

    # additional ``kwargs`` for more informative report
    paradigm=paradigm.__dict__,
    TR=TR,
    n_scans=n_scans,
    hfcut=hfcut,
    frametimes=frametimes,
    drift_model=drift_model,
    hrf_model=hrf_model,
    )

# shutdown main report page
base_reporter.ProgressReport().finish_dir(output_dir)

print "\r\nStatistic report written to %s\r\n" % stats_report_filename
