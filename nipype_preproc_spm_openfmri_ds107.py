"""
:Module: nipype_preproc_spm_openfmri_ds107
:Synopsis: Preprocessing Openfmri ds107
:Author: dohmatob elvis dopgima

"""

# standard imports
import os
import glob
import sys

# import spm preproc utilities
import nipype_preproc_spm_utils

# misc
from external.nisl.datasets import unzip_nii_gz

DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""

# wildcard defining directory structure
subject_id_wildcard = "sub*"

# subject ids to ignore
ignore_these_subjects = ['sub003',  # garbage anat image
                         ]

# sessions we're interested in
SESSION_IDs = ["task001_run001", "task001_run002"]
SESSION_IDs.sort()

# DARTEL ?
DO_DARTEL = False


def main(DATA_DIR, OUTPUT_DIR):
    """
    returns list of Bunch objects with fields anat, func, and subject_id
    for each preprocessed subject

    """

    # glob for subject ids
    subject_ids = [os.path.basename(x)
                   for x in glob.glob(
            os.path.join(DATA_DIR, subject_id_wildcard))]

    # always sort such thinks ;)
    subject_ids.sort()

    # callback for determining which subject's to skip
    def ignore_subject_id(subject_id):
        return subject_id in ignore_these_subjects

    # producer subject data
    def subject_factory():
        for subject_id in subject_ids:
            if ignore_subject_id(subject_id):
                continue

            # construct subject data structure
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.session_id = SESSION_IDs
            subject_data.subject_id = subject_id
            subject_data.func = []

            # orientation meta-data for sub013 is garbage
            if subject_id in ['sub013']:
                subject_data.bad_orientation = True

            # glob for bold data
            for session_id in subject_data.session_id:
                bold_dir = os.path.join(
                    DATA_DIR,
                    "%s/BOLD/%s" % (subject_id, session_id))

                # extract .nii.gz to .ni
                unzip_nii_gz(bold_dir)

                # glob bold data proper
                func = glob.glob(
                    os.path.join(
                        DATA_DIR,
                        "%s/BOLD/%s/bold.nii" % (
                            subject_id, session_id)))[0]
                subject_data.func.append(func)

            # glob for anatomical data
            anat_dir = os.path.join(
                DATA_DIR,
                "%s/anatomy" % subject_id)

            # extract .nii.gz to .ni
            unzip_nii_gz(anat_dir)

            # glob anatomical data proper
            subject_data.anat = glob.glob(
                os.path.join(
                    DATA_DIR,
                    "%s/anatomy/highres001_brain.nii" % subject_id))[0]

            # set subject output dir (all calculations for
            # this subject go here)
            subject_data.output_dir = os.path.join(
                    OUTPUT_DIR,
                    subject_id)

            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(OUTPUT_DIR,
                                   "_report.html")
    return nipype_preproc_spm_utils.do_group_preproc(
        subject_factory(),
        output_dir=OUTPUT_DIR,
        # delete_orientation=True,
        do_dartel=DO_DARTEL,
        do_cv_tc=False,
        # do_report=False,
        # do_export_report=True,
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename
        )

if __name__ == '__main__':
    # sanitize cmd-line input
    if len(sys.argv)  < 3:
        print ("\r\nUsage: source /etc/fsl/4.1/fsl.sh; python %s "
               "<path_to_openfmri_ds107_folder> <output_dir>\r\n"
               ) % sys.argv[0]
        print ("Example:\r\nsource /etc/fsl/4.1/fsl.sh; python %s "
               "/vaporific/edohmato/datasets/openfmri/ds107/ "
               "/vaporific/edohmato/pypreprocess_runs/openfmri/ds107"
               ) % sys.argv[0]
        sys.exit(1)

    DATA_DIR = os.path.abspath(sys.argv[1])

    OUTPUT_DIR = os.path.abspath(sys.argv[2])
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if len(sys.argv) > 3:
        subject_id_wildcard = sys.argv[3]

    # collect preprocessed results (one per subject)
    results = main(DATA_DIR, OUTPUT_DIR)

    # start level 1 analysis here ;)
    pass

    print results
