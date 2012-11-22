"""
:Module: nipype_preproc_spm_haxby
:Synopsis: SPM use-case for preprocessing HAXBY dataset
(this is just a quick-and-dirty POC)
:Author: dohmatob elvis dopgima

XXX TODO: document this according to numpy/spinx standards
XXX TODO: re-factor the code (use unittesting)
XXX TODO: more  preprocessing checks (coregistration step, etc.)
XXX TODO: over-all testing (nose ?, see with GV & BT)

"""

# standard imports
import os
import glob

# helper imports
from nisl.datasets import fetch_haxby, unzip_nii_gz

# import spm preproc utilities
import nipype_preproc_spm_utils

# QA imports
from check_preprocessing import *
import markup
from report_utils import *
import time

# set data dir
if not 'DATA_DIR' in os.environ:
    raise IOError("DATA_DIR is not in your environ; export it!")
DATA_DIR = os.environ['DATA_DIR']

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

if __name__ == '__main__':
    # fetch HAXBY dataset
    haxby_data = fetch_haxby(data_dir=DATA_DIR,
                             subject_ids=["subj4", "subj2", "subj3"])

    # producer
    def subject_factory():
        for subject_id, subject_data in haxby_data.iteritems():
            # pre-process data for all subjects
            subject_dir = subject_data["subject_dir"]
            unzip_nii_gz(subject_dir)
            anat_image = subject_data["anat"].replace(".gz", "")
            fmri_images = subject_data["bold"].replace(".gz", "")
            yield subject_id, subject_dir, anat_image, fmri_images,\
                "haxbyby2001"

    nipype_preproc_spm_utils.do_group_preproc(
        subject_factory(),
        dataset_description=DATASET_DESCRIPTION,
        report_filename=os.path.join(DATA_DIR, "haxby_preproc_report.html"))
