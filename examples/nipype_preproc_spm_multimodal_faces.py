"""
:Module: nipype_preproc_spm_multimodal_faces
Synopsis: Minimal script for preprocessing single-subject data +
GLM with nipy
Author: dohmatob elvis dopgima elvis[dot]dohmatob[at]inria[dot]fr

"""

# standard imports
import sys
import os
import time
import nibabel
import numpy as np
import scipy.io

# imports for GLM buusiness
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel

# pypreprocess imports
from pypreprocess.datasets import fetch_spm_multimodal_fmri_data
from pypreprocess.reporting.base_reporter import ProgressReport
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData)

# file containing configuration for preprocessing the data
this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
jobfile = os.path.join(this_dir, "multimodal_faces_preproc.ini")

# set dataset dir
if len(sys.argv) > 1:
    dataset_dir = sys.argv[1]
else:
    dataset_dir = os.path.join(this_dir, "spm_multimodal_faces")

# fetch spm multimodal_faces data
subject_data = fetch_spm_multimodal_fmri_data(dataset_dir)

# preprocess the data
subject_id = "sub001"
subject_data = SubjectData(output_dir=os.path.join(
        dataset_dir, "pypreprocess", subject_id),
                           subject_id=subject_id,
                           func=[subject_data.func1, subject_data.func2],
                           anat=subject_data.anat,
                           trials_ses1=subject_data.trials_ses1,
                           trials_ses2=subject_data.trials_ses2)
subject_data = do_subjects_preproc([subject_data], realign=True,
                                  coregister=True, segment=True,
                                  normalize=True)[0]

# experimental paradigm meta-params
stats_start_time = time.ctime()
tr = 2.
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
hfcut = 128.

# make design matrices
first_level_effects_maps = []
mask_images = []
design_matrices = []
for x in xrange(2):
    if not os.path.exists(subject_data.output_dir):
        os.makedirs(subject_data.output_dir)

    # build paradigm
    n_scans = len(subject_data.func[x])
    timing = scipy.io.loadmat(getattr(subject_data, "trials_ses%i" % (x + 1)),
                              squeeze_me=True, struct_as_record=False)

    faces_onsets = timing['onsets'][0].ravel()
    scrambled_onsets = timing['onsets'][1].ravel()
    onsets = np.hstack((faces_onsets, scrambled_onsets))
    onsets *= tr  # because onsets were reporting in 'scans' units
    conditions = ['faces'] * len(faces_onsets) + ['scrambled'] * len(
        scrambled_onsets)
    paradigm = EventRelatedParadigm(conditions, onsets)

    # build design matrix
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    design_matrix = make_dmtx(
        frametimes,
        paradigm, hrf_model=hrf_model,
        drift_model=drift_model, hfcut=hfcut,
        add_reg_names=['tx', 'ty', 'tz', 'rx', 'ry', 'rz'],
        add_regs=np.loadtxt(subject_data.realignment_parameters[x])
        )

    design_matrices.append(design_matrix)

# specify contrasts
contrasts = {}
n_columns = len(design_matrix.names)
for i in xrange(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]
              ] = np.eye(n_columns)[2 * i]

# more interesting contrasts
contrasts['faces-scrambled'] = contrasts['faces'
                                         ] - contrasts['scrambled']
contrasts['scrambled-faces'] = -contrasts['faces-scrambled']
contrasts['effects_of_interest'] = contrasts['faces'
                                             ] + contrasts['scrambled']

# fit GLM
print 'Fitting a GLM (this takes time)...'
fmri_glm = FMRILinearModel([nibabel.concat_images(x)
                            for x in subject_data.func],
                           [design_matrix.matrix
                            for design_matrix in design_matrices],
                           mask='compute'
                           )
fmri_glm.fit(do_scaling=True, model='ar1')

# save computed mask
mask_path = os.path.join(subject_data.output_dir,
                         "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)
mask_images.append(mask_path)

# compute contrasts
z_maps = {}
effects_maps = {}
for contrast_id, contrast_val in contrasts.iteritems():
    print "\tcontrast id: %s" % contrast_id
    z_map, t_map, effects_map, var_map = fmri_glm.contrast(
        [contrast_val] * 2,
        con_id=contrast_id,
        output_z=True,
        output_stat=True,
        output_effects=True,
        output_variance=True
        )

    # store stat maps to disk
    for map_type, out_map in zip(['z', 't', 'effects', 'variance'],
                              [z_map, t_map, effects_map, var_map]):
        map_dir = os.path.join(
            subject_data.output_dir, '%s_maps' % map_type)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(
            map_dir, '%s.nii.gz' % contrast_id)
        print "\t\tWriting %s ..." % map_path
        nibabel.save(out_map, map_path)

        # collect zmaps for contrasts we're interested in
        if map_type == 'z':
            z_maps[contrast_id] = map_path
        if map_type == 'effects':
            effects_maps[contrast_id] = map_path

# do stats report
anat_img = nibabel.load(subject_data.anat)
stats_report_filename = os.path.join(subject_data.output_dir,
                                     "reports",
                                     "report_stats.html")
generate_subject_stats_report(
    stats_report_filename,
    contrasts,
    z_maps,
    fmri_glm.mask,
    threshold=2.3,
    cluster_th=15,
    anat=anat_img.get_data(),
    anat_affine=anat_img.get_affine(),
    design_matrices=design_matrix,
    subject_id="sub001",
    start_time=stats_start_time,
    title="GLM for subject %s" % subject_data.subject_id,

    # additional ``kwargs`` for more informative report
    paradigm=paradigm.__dict__,
    TR=tr,
    n_scans=n_scans,
    hfcut=hfcut,
    frametimes=frametimes,
    drift_model=drift_model,
    hrf_model=hrf_model,
    )

ProgressReport().finish_dir(subject_data.output_dir)
print "Statistic report written to %s\r\n" % stats_report_filename
