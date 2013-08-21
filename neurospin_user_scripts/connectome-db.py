"""
:Author: dohmatob elvis dopgima <gmdopp@gmail.com>

"""

import os
import sys
import glob

# import spm preproc utilities
sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(sys.argv[0]))))
import nipype_preproc_spm_utils

# data fetching imports
from external.nilearn.datasets import fetch_nyu_rest
from datasets_extras import unzip_nii_gz

# session ids we're interested in
SESSIONS = ['tfMRI_LANGUAGE', 'tfMRI_MOTOR']
DIRECTIONS = ['LR', 'RL']

# main code follows
if __name__ == '__main__':
    # sanitize input
    if len(sys.argv) < 3:
        print ("\r\nUsage: python %s "
               "<connection_db_data_dir> <output_dir>") % sys.argv[0]
        sys.exit(-1)

    data_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    skipped_subjects = []

    def subject_factory():
        """
        Factory for subject data.

        """

        def _skip_subject(subject_id, reason):
            skipped_subjects.append((subject_id, reason))
            print "Skipping subject %s: %s..." % (subject_id, reason)

        subject_stuff = sorted([(x, os.path.basename(x))
                                for x in glob.glob(os.path.join(
                        data_dir, "*")) if os.path.isdir(x)])

        for subject_data_dir, subject_id in subject_stuff:
            # instantiate subject data object
            subject_data = nipype_preproc_spm_utils.SubjectData()

            # misc
            subject_data.subject_id = subject_id
            subject_data.session_id = SESSIONS
            subject_data.output_dir = os.path.join(output_dir, subject_id)

            # glob anat
            try:
                subject_data.anat = glob.glob(os.path.join(
                        subject_data_dir,
                        "Structural/T1*_acpc_dc_restore_brain.nii.gz"))[0]
            except IndexError:
                _skip_subject(subject_id, "missing anat data")
                continue

            # glob for subject func data
            try:
                subject_data.func = [[sorted(glob.glob(os.path.join(
                                    subject_data_dir,
                                    "%s/unprocessed/3T/*/*%s_%s.nii.gz" % (
                                        session_id, session_id,
                                        direction))))[0]
                                      for direction in DIRECTIONS][
                        0]  # XXX only LR direction for now
                                     for session_id in SESSIONS]
            except IndexError:
                _skip_subject(subject_id, "some functional data is missing")
                continue

            # yield subject data
            yield subject_data

    if len(skipped_subjects):
        print("\r\nWarning: Skipped subjects:\r\n\t%s\r\n" % "\r\n\t".join(
                ["%s: %s" % (subject_id, reason)
                 for subject_id, reason in skipped_subjects]))

    # preprocess the subjects proper
    nipype_preproc_spm_utils.do_subjects_preproc(subject_factory(),
                                                 output_dir=output_dir,
                                                 dataset_id='CONNECTOME-DB',
                                                 #do_coreg=False,
                                                 func_to_anat=True,

                                                 # no normalization, etc.
                                                 do_segment=False,
                                                 do_normalize=False,
                                                 )
