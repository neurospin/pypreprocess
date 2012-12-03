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

# set data dir
if not 'DATA_DIR' in os.environ:
    raise IOError("DATA_DIR is not in your environ; export it!")
DATA_DIR = os.environ['DATA_DIR']

# set output dir (never pollute data dir!!!)
OUTPUT_DIR = os.getcwd()
if 'OUTPUT_DIR' in os.environ:
    OUTPUT_DIR = os.environ['OUTPUT_DIR']
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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
    haxby_data = fetch_haxby(data_dir=DATA_DIR)

    # producer
    def subject_factory():
        for subject_id, _subject_data in haxby_data.iteritems():
            # pre-process data for all subjects
            subject_data = nipype_preproc_spm_utils.SubjectData()
            session_id = "haxby2001"
            subject_data.subject_id = subject_id
            unzip_nii_gz(_subject_data['subject_dir'])
            subject_data.anat = _subject_data["anat"].replace(".gz", "")
            subject_data.func = _subject_data["bold"].replace(".gz", "")
            subject_data.output_dir = os.path.join(
                os.path.join(OUTPUT_DIR, session_id),
                subject_id)
            if not os.path.exists(subject_data.output_dir):
                os.makedirs(subject_data.output_dir)

            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(OUTPUT_DIR,
                                   "_report.html")

    nipype_preproc_spm_utils.do_group_preproc(
        subject_factory(),
        do_realign=False,
        do_coreg=False,
        do_cv_tc=False,
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename)
