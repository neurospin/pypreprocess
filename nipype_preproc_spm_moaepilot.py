"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing ABIDE MAxMun rest dataset
:Author: dohmatob elvis dopgima

"""

# standard imports
import os
import glob
import sys

# import spm preproc utilities
import nipype_preproc_spm_utils

DATASET_DESCRIPTION = """\
<p>MoAEpilot SPM auditory dataset</p>\
"""

if __name__ == '__main__':
    # sanitize cmd-line input
    if len(sys.argv)  < 3:
        print "\r\nUsage: python %s <data_dir> <output_dir>\r\n" \
            % sys.argv[0]
        sys.exit(1)

    DATA_DIR = os.path.abspath(sys.argv[1])

    OUTPUT_DIR = os.path.abspath(sys.argv[2])
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    def subject_factory():
        subject_id = "sub001"
        subject_data = nipype_preproc_spm_utils.SubjectData()
        subject_data.subject_id = subject_id
        subject_data.session_id = 'UNKNOWN_SESSION'

        subject_data.func = glob.glob(
            os.path.join(
                DATA_DIR,
                "fM00223/*.img"))

        subject_data.anat = glob.glob(
            os.path.join(
                DATA_DIR,
                "sM00223/sM00223_002.img"))

        subject_data.output_dir = os.path.join(
            os.path.join(
                OUTPUT_DIR, subject_data.session_id),
            subject_id)

        yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(OUTPUT_DIR,
                                   "_report.html")
    nipype_preproc_spm_utils.do_group_preproc(
        subject_factory(),
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename)
