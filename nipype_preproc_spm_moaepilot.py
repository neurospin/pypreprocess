"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing SPM auditory data
:Author: dohmatob elvis dopgima

"""

# standard imports
import os
import sys

# import spm preproc utilities
import nipype_preproc_spm_utils

from nisl.datasets import fetch_spm_auditory_data

DATASET_DESCRIPTION = """\
<p>MoAEpilot <a href="http://www.fil.ion.ucl.ac.uk/spm/data/auditory/">\
SPM auditory dataset</a>.</p>\
"""

if __name__ == '__main__':
    # sanitize cmd-line input
    if len(sys.argv)  < 3:
        print ("\r\nUsage: python %s <spm_auditory_MoAEpilot_dir>"
               " <output_dir>\r\n") % sys.argv[0]
        sys.exit(1)

    DATA_DIR = os.path.abspath(sys.argv[1])

    OUTPUT_DIR = os.path.abspath(sys.argv[2])
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    def subject_factory():
        '''
        Producer for subject data.

        '''

        # fetch spm auditory data
        _subject_data = fetch_spm_auditory_data(DATA_DIR)

        subject_data = nipype_preproc_spm_utils.SubjectData()
        subject_data.func = _subject_data["func"]
        subject_data.anat = _subject_data["anat"]
        subject_data.output_dir = os.path.join(
            os.path.join(
                OUTPUT_DIR, subject_data.session_id),
            subject_data.subject_id)

        yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(OUTPUT_DIR,
                                   "_report.html")
    nipype_preproc_spm_utils.do_group_preproc(
        subject_factory(),
        dataset_description=DATASET_DESCRIPTION,
        do_export_report=True,
        report_filename=report_filename)
