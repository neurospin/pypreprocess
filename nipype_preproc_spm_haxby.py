"""
:Module: nipype_preproc_spm_haxby
:Synopsis: SPM use-case for preprocessing HAXBY 2001 dataset
:Author: dohmatob elvis dopgima

"""

# standard imports
import os
import sys

# data-grabbing imports
from datasets_extras import unzip_nii_gz, fetch_haxby

# import spm preproc utilities
import nipype_preproc_spm_utils

# DARTEL ?
DO_DARTEL = False

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

"""sanitize cmd-line"""
if len(sys.argv) < 3:
    print "Usage: python %s <haxby_dir> <output_dir>" % sys.argv[0]
    print ("Example:\r\npython %s ~/CODE/datasets/haxby"
           " ~/CODE/FORKED/pypreprocess/haxby_runs") % sys.argv[0]
    sys.exit(1)

"""set dataset dir"""
DATA_DIR = sys.argv[1]

"""set output dir"""
OUTPUT_DIR = sys.argv[2]
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# fetch HAXBY dataset
subject_ids = ["subj1", "subj2", "subj3", "subj4", "subj5"]
haxby_data = fetch_haxby(
    data_dir=DATA_DIR,
    subject_ids=subject_ids,
    n_jobs=len(subject_ids))


def subject_factory():
    """producer for subject (input) data"""
    for subject_id, sd in haxby_data.iteritems():
        subject_data = nipype_preproc_spm_utils.SubjectData()
        subject_data.session_id = "haxby2001"
        subject_data.subject_id = subject_id
        unzip_nii_gz(sd.subject_dir)
        subject_data.anat = sd.anat.replace(".gz", "")
        subject_data.func = sd.bold.replace(".gz", "")
        subject_data.output_dir = os.path.join(
            OUTPUT_DIR, subject_data.subject_id)

        yield subject_data

"""do preprocessing proper"""
report_filename = os.path.join(OUTPUT_DIR,
                               "_report.html")

results = nipype_preproc_spm_utils.do_subjects_preproc(
    subject_factory(),
    output_dir=OUTPUT_DIR,
    do_realign=False,
    do_coreg=False,
    do_dartel=DO_DARTEL,
    do_cv_tc=False,
    dataset_description=DATASET_DESCRIPTION,
    report_filename=report_filename)
