"""
Author: DOHMATOB Elvis Dopgima elvis[dot]dohmatob[at]inria[dot]fr
Synopsis: Preprocessing of HAXBY 2001 dataset
"""

import os
from nilearn.datasets import fetch_haxby
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData)

# DARTEL ?
DARTEL = False

DATASET_DESCRIPTION = """\
This is a block-design fMRI dataset from a study on face and object\
 representation in human ventral temporal cortex. It consists of 6 subjects\
 with 12 runs per subject. In each run, the subjects passively viewed \
greyscale images of eight object categories, grouped in 24s blocks separated\
 by rest periods. Each image was shown for 500ms and was followed by a 1500ms\
 inter-stimulus interval. Full-brain fMRI data were recorded with a volume \
repetition time of 2.5s, thus, a stimulus block was covered by roughly 9 \
volumes.

Get full description <a href="http://dev.pymvpa.org/datadb/haxby2001.html">\
here</a>.\
"""

# fetch HAXBY dataset
N_SUBJECTS = 2
haxby_data = fetch_haxby(subjects=N_SUBJECTS)

# set output dir
OUTPUT_DIR = os.path.join(os.path.dirname(haxby_data.mask),
                          "haxby_runs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# get subject data
subjects = []
for subject_id in set([os.path.basename(os.path.dirname(x))
                       for x in haxby_data.func]):
    # instantiate subject_data object
    subject_data = SubjectData()
    subject_data.subject_id = subject_id
    subject_data.session_id = "haxby2001"

    # set func
    subject_data.func = [x for x in haxby_data.func if subject_id in x]

    assert len(subject_data.func) == 1
    subject_data.func = subject_data.func[0]

    # set anat
    subject_data.anat = [x for x in haxby_data.anat if subject_id in x]
    assert len(subject_data.anat) == 1
    subject_data.anat = subject_data.anat[0]

    # set subject output directory
    subject_data.output_dir = os.path.join(OUTPUT_DIR,
                                           subject_data.subject_id)

    # add this subject to list
    subjects.append(subject_data)

# do preprocessing proper
results = do_subjects_preproc(subjects, output_dir=OUTPUT_DIR,
                              dataset_id="HAXBY 2001", realign=False,
                              coregister=False, dartel=DARTEL,
                              tsdiffana=False,
                              dataset_description=DATASET_DESCRIPTION)
