"""
Author: DOHMATOB Elvis Dopgima elvis[dot]dohmatob[at]inria[dot]fr
Synopsis: Demo script for nipy's GLM and inference + reporting
on FSL's FEEDS fMRI single-subject example data
"""

import os
import numpy as np
from pandas import DataFrame
from pypreprocess.external.nistats.design_matrix import (make_design_matrix,
                                                         check_design_matrix)
from pypreprocess.external.nistats.glm import FMRILinearModel
import nibabel
import time
from pypreprocess.nipype_preproc_spm_utils import (SubjectData,
                                                   do_subjects_preproc)
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.reporting.base_reporter import ProgressReport
from pypreprocess.datasets import fetch_fsl_feeds
from pypreprocess.io_utils import compute_mean_3D_image

"""MISC"""
DATASET_DESCRIPTION = "FSL FEEDS example data (single-subject)"

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
paradigm = DataFrame({'name': conditions, 'onset': onset,
                      'duration': duration})
frametimes = np.linspace(0, (n_scans - 1) * TR, n_scans)
maximum_epoch_duration = max(EV1_epoch_duration, EV2_epoch_duration)
hfcut = 1.5 * maximum_epoch_duration  # why ?

"""construct design matrix"""
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
design_matrix = make_design_matrix(frame_times=frametimes,
                                   paradigm=paradigm,
                                   hrf_model=hrf_model,
                                   drift_model=drift_model,
                                   period_cut=hfcut)

"""fetch input data"""
_subject_data = fetch_fsl_feeds()
subject_data = SubjectData()
subject_data.subject_id = "sub001"
subject_data.func = _subject_data.func
subject_data.anat = _subject_data.anat

output_dir = os.path.join(_subject_data.data_dir, "pypreprocess_output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
subject_data.output_dir = os.path.join(
    output_dir, subject_data.subject_id)



"""preprocess the data"""
results = do_subjects_preproc(
    [subject_data],
    output_dir=output_dir,
    dataset_id="FSL FEEDS single-subject",
    dataset_description=DATASET_DESCRIPTION,
    do_shutdown_reloaders=False,
    )

"""collect preprocessed data"""
fmri_files = results[0]['func']
anat_file = results[0]['anat']

"""specify contrasts"""
_, matrix, names = check_design_matrix(design_matrix)
contrasts = {}
n_columns = len(names)
I = np.eye(len(names))
for i in xrange(2):
    contrasts['%s' % names[2 * i]] = I[2 * i]

"""more interesting contrasts"""
contrasts['EV1>EV2'] = contrasts['EV1'] - contrasts['EV2']
contrasts['EV2>EV1'] = contrasts['EV2'] - contrasts['EV1']
contrasts['effects_of_interest'] = contrasts['EV1'] + contrasts['EV2']

"""fit GLM"""
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FMRILinearModel(fmri_files, matrix, mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

"""save computed mask"""
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)

# compute bg unto which activation will be projected
mean_fmri_files = compute_mean_3D_image(fmri_files)
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
reports_dir = os.path.join(subject_data.output_dir, "reports")
stats_report_filename = os.path.join(reports_dir, "report_stats.html")
contrasts = dict((contrast_id, contrasts[contrast_id])
                 for contrast_id in z_maps.keys())
generate_subject_stats_report(
    stats_report_filename,
    contrasts,
    z_maps,
    fmri_glm.mask,
    design_matrices=[design_matrix],
    subject_id=subject_data.subject_id,
    anat=anat_file,
    cluster_th=50,  # we're only interested in this 'large' clusters
    start_time=stats_start_time,

    # additional ``kwargs`` for more informative report
    paradigm=paradigm,
    TR=TR,
    n_scans=n_scans,
    hfcut=hfcut,
    frametimes=frametimes,
    drift_model=drift_model,
    hrf_model=hrf_model,
    slicer='z'
    )

# shutdown main report page
ProgressReport().finish_dir(output_dir)

print "\r\nStatistic report written to %s\r\n" % stats_report_filename
