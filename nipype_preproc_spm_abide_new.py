"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing ABIDE auditory rest dataset
:Author: dohmatob elvis dopgima

"""

# standard imports
import os
import sys

# import spm preproc utilities
import nipype_preproc_spm_utils
from external.nisl.more_datasets import fetch_abide

DATASET_DESCRIPTION = """\
<p>ABIDE rest auditory dataset.</p>\
"""

if __name__ == '__main__':
    # sanitize cmd-line input
    if len(sys.argv) < 2:
        print ("\r\nUsage: source /etc/fsl/4.1/fsl.sh; python %s "
               "<output_dir>\r\n") % sys.argv[0]
        print ("Example:\r\nsource /etc/fsl/4.1/fsl.sh; python %s "
               "/volatile/home/aa013911/DED/ABIDE_runs") % sys.argv[0]
        sys.exit(1)

    OUTPUT_DIR = os.path.abspath(sys.argv[2])
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    abide = fetch_abide(SITE_ID='UM_1')

    # producer for MaxMun subjects
    def subject_factory():
        for i in range(len(abide['func'])):
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.subject_id = abide['pheno']['SUB_ID'][i]

            subject_data.func = abide['func'][i]
            subject_data.anat = abide['anat'][i]
            subject_data.session_id = 'abide'

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
        delete_orientation=True,
        do_export_report=False,
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename)
