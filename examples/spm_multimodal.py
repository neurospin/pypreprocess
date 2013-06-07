import glob
import sys
import os
import pylab as pl
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
import nibabel
import time

warning = ("%s: THIS SCRIPT MUST BE RUN FROM ITS PARENT "
           "DIRECTORY!") % sys.argv[0]
banner = "#" * len(warning)
separator = "\r\n\t"

print separator.join(['', banner, warning, banner, ''])

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])

sys.path.append(PYPREPROCESS_DIR)

import nipype_preproc_spm_utils
import reporting.glm_reporter as glm_reporter

DATA_DIR = "/home/elvis/Downloads/alex_spm/"
OUTPUT_DIR = "spm_multimodal_runs"

# fetch the data
subject_data = nipype_preproc_spm_utils.SubjectData()

subject_data.subject_id = "sub001"
subject_data.session_id = ["Session1", "Session2"]

subject_data.func = [sorted(glob.glob(os.path.join(
                DATA_DIR,
                "fMRI/%s/fMETHODS-*.img" % s)))
                     for s in subject_data.session_id]

subject_data.anat = os.path.join(DATA_DIR, "sMRI/smri.img")

subject_data.output_dir = os.path.join(OUTPUT_DIR,
                                       subject_data.subject_id)


"""preprocess the data"""
results = nipype_preproc_spm_utils.do_subjects_preproc(
    [subject_data],
    output_dir=OUTPUT_DIR,
    fwhm=[8, 8, 8],
    dataset_id="SPM MULTIMODAL (see @alex)",
    do_shutdown_reloaders=False,
    )
