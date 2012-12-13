"""
:Module: nipype_preproc_spm_haxby
:Synopsis: SPM use-case for preprocessing HAXBY dataset
(this is just a quick-and-dirty POC)
:Author: dohmatob elvis dopgima

XXX TODO: document this according to numpy/spinx standards
XXX TODO: re-factor the code (use unittesting)
XXX TODO: more  preprocessing checks (coregistration step, etc.)
XXX TODO: over-all testing (nose ?, see with GV & BT)

Example cmd-line run (there are no breaks between the following lines):
DATA_DIR=~/haxby_data N_JOBS=-1 \
OUTPUT_DIR=~/haxby_runs \
python nipype_preproc_spm_haxby.py

"""

# standard imports
import os

# helper imports
from external.nisl.datasets import fetch_haxby, _uncompress_file

# import spm preproc utilities
import nipype_preproc_spm_utils

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
    n_subjects = 5
    haxby_data = fetch_haxby(n_subjects=n_subjects)

    # producer
    def subject_factory():
        for i in range(n_subjects):
            # pre-process data for all subjects
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.session_id = "haxby2001"
            subject_data.subject_id = 'subj%d' % (i + 1)
            _uncompress_file(haxby_data['anat'][i], delete_archive=False)
            _uncompress_file(haxby_data['func'][i], delete_archive=False)
            subject_data.anat = haxby_data['anat'][i].replace(".gz", "")
            subject_data.func = haxby_data['func'][i].replace(".gz", "")
            subject_data.output_dir = os.path.join(
                os.path.join(OUTPUT_DIR, subject_data.session_id),
                'subj%d' % (i + 1))

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
