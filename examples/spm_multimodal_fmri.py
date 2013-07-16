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
import nibabel
import scipy.io
import time
import sys
import os
from collections import namedtuple

warning = ("%s: THIS SCRIPT MUST BE RUN FROM ITS PARENT "
           "DIRECTORY!") % sys.argv[0]
banner = "#" * len(warning)
separator = "\r\n\t"

print separator.join(['', banner, warning, banner, ''])

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])
sys.path.append(PYPREPROCESS_DIR)

import reporting.glm_reporter as glm_reporter
from external.nilearn.datasets import fetch_spm_multimodal_fmri_data
from algorithms.registration.spm_realign import MRIMotionCorrection
from algorithms.slice_timing.spm_slice_timing import fMRISTC

# datastructure for subject data
SubjectData = namedtuple('SubjectData',
                         'subject_id session_id func anat output_dir')

# set data and output paths (change as you will)
DATA_DIR = "spm_multimodal_fmri"
OUTPUT_DIR = "spm_multimodal_runs"
if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
if len(sys.argv) > 2:
    OUTPUT_DIR = sys.argv[2]

print "\tDATA_DIR: %s" % DATA_DIR
print "\tOUTPUT_DIR: %s" % OUTPUT_DIR
print

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DO_REPORT = True

# fetch the data
_subject_data = fetch_spm_multimodal_fmri_data(DATA_DIR)
subject_id = "sub001"
subject_data = SubjectData(subject_id=subject_id,
                           session_id=["Session1", "Session2"],
                           func=[_subject_data.func1, _subject_data.func2],
                           anat=_subject_data.anat,
                           output_dir=os.path.join(OUTPUT_DIR, subject_id)
                           )

# STC
fmristc = fMRISTC(slice_order='descending', interleaved=False)
stc_output = []
for j in xrange(len(subject_data.session_id)):
    fmristc.fit(raw_data=subject_data.func[j])
    fmristc.transform()
    stc_output.append(fmristc.get_last_output_data())

# Motion Correction
mrimc = MRIMotionCorrection(n_sessions=len(subject_data.session_id))
mrimc.fit([[nibabel.Nifti1Image(stc_output[j][..., t],
                                nibabel.load(
                    subject_data.func[j][t]).get_affine())
            for t in xrange(len(subject_data.func[j]))]
           for j in xrange(len(subject_data.session_id))]
          )
mrimc_output = mrimc.transform(os.path.join(subject_data.output_dir,
                                            "preproc"),
                               reslice=True,
                               concat=True,
                               )

# collect preprocessed data
fmri_files = mrimc_output['realigned_files']
rp_filenames = mrimc_output['rp_filenames']
anat_img = nibabel.four_to_three(nibabel.load(fmri_files[0]))[0]

# experimental paradigm meta-params
stats_start_time = time.ctime()
tr = 2.
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
hfcut = 128.

# make design matrices
design_matrices = []
for x in xrange(2):
    n_scans = nibabel.load(fmri_files[x]).shape[-1]

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
    design_matrix = make_dmtx(
        frametimes,
        paradigm, hrf_model=hrf_model,
        drift_model=drift_model, hfcut=hfcut,
        add_reg_names=['tx', 'ty', 'tz', 'rx', 'ry', 'rz'],
        add_regs=np.loadtxt(rp_filenames[x])
        )

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
fmri_glm = FMRILinearModel(fmri_files,
                           [dmat.matrix for dmat in design_matrices],
                           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

# save computed mask
mask_path = os.path.join(OUTPUT_DIR, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)

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
        nibabel.save(out_map, map_path)

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
        cluster_th=15,  # we're only interested in this 'large' clusters
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

    from reporting.base_reporter import ProgressReport
    ProgressReport().finish_dir(OUTPUT_DIR)

    print "\r\nStatistic report written to %s\r\n" % stats_report_filename
