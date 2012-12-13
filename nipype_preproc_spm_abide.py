"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing ABIDE auditory rest dataset
:Author: dohmatob elvis dopgima

"""

# standard imports
import os
import glob
import sys

# import spm preproc utilities
import nipype_preproc_spm_utils

DATASET_DESCRIPTION = """\
<p>ABIDE rest auditory dataset.</p>\
"""

# XXX change this to */* if preprocessing all ABIDE subjects
subject_id_wildcard = "MaxMun_*/MaxMun_*"

if __name__ == '__main__':
    # sanitize cmd-line input
    if len(sys.argv)  < 3:
        print ("\r\nUsage: source /etc/fsl/4.1/fsl.sh; python %s "
               "<path_to_ABIDE_folder> <output_dir>\r\n") % sys.argv[0]
        print ("Example:\r\nsource /etc/fsl/4.1/fsl.sh; python %s ~/ABIDE "
               "/volatile/home/aa013911/DED/ABIDE_runs") % sys.argv[0]
        sys.exit(1)

    ABIDE_DIR = os.path.abspath(sys.argv[1])

    OUTPUT_DIR = os.path.abspath(sys.argv[2])
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # glob for MaxMun subject_ids
    subject_ids = [os.path.basename(x)
                   for x in glob.glob(
            os.path.join(ABIDE_DIR, subject_id_wildcard))]

    # producer for MaxMun subjects
    def subject_factory():
        for subject_id in subject_ids:
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.subject_id = subject_id

            subject_data.func = glob.glob(
                os.path.join(
                    ABIDE_DIR,
                    "%s/%s/scans/rest/resources/NIfTI/files/rest.nii" % (
                        subject_id, subject_id)))

            subject_data.anat = glob.glob(
                os.path.join(
                    ABIDE_DIR,
                    "%s/%s/scans/anat/resources/NIfTI/files/mprage.nii" % (
                        subject_id, subject_id)))

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
        delete_orientation=True,
        do_export_report=False,
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename)
