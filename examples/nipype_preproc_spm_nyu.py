"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing NYU rest dataset
(this is just a quick-and-dirty POC)
:Author: dohmatob elvis dopgima

XXX TODO: document this according to numpy/spinx standards
XXX TODO: re-factor the code (use unittesting)
XXX TODO: over-all testing (nose ?, see with GV & BT)

Example cmd-line run (there are no breaks between the following lines):
SPM_DIR=~/spm8 MATLAB_EXEC=/usr/local/MATLAB/R2011a/bin/matlab \
DATA_DIR=~/CODE/datasets/nyu_data/nyu_rest/nyu_rest/ N_JOBS=-1 \
OUTPUT_DIR=~/CODE/FORKED/pypreprocess/nyu_runs \
python nipype_preproc_spm_nyu.py

"""

# standard imports
import os
import sys

# import spm preproc utilities
sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(sys.argv[0]))))
import nipype_preproc_spm_utils

# data fetching imports
from external.nilearn.datasets import fetch_nyu_rest
from datasets_extras import unzip_nii_gz

DATASET_DESCRIPTION = """\
<p>The NYU CSC TestRetest resource includes EPI-images of 25 participants
gathered during rest as well as anonymized anatomical images of the \
same participants.</p>

<p>The resting-state fMRI images were collected on several occasions:<br>
1. the first resting-state scan in a scan session<br>
2. 5-11 months after the first resting-state scan<br>
3. about 30 (< 45) minutes after 2.</p>

<p>Get full description <a href="http://www.nitrc.org/projects/nyu_trt\
/">here</a>.</p>
"""

# use DARTEL for normalization or not ?
DARTEL = False

# session ids we're interested in
SESSIONS = [1, 2, 3]

# bad subjects
BAD_SUBJECTS = ["sub39529",  # in func, a couple of volumes are blank!
                "sub45463",  # anat has bad header info
                ]

# main code follows
if __name__ == '__main__':
    # sanitize input
    if len(sys.argv) < 3:
        print ("\r\nUsage: source /etc/fsl/4.1/fsl.sh; python %s "
               "<NYU_data_dir> <output_dir>") % sys.argv[0]
        print ("Example: source /etc/fsl/4.1/fsl.sh; python %s "
               "~/CODE/datasets/nyu_data nyu_runs\r\n") % sys.argv[0]
        sys.exit(-1)

    OUTPUT_DIR = os.path.abspath(sys.argv[2])
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # fetch data
    nyu_data = fetch_nyu_rest(data_dir=sys.argv[1], sessions=[1, 2, 3])

    # subject data factory
    def subject_factory(session_output_dir, session):
        session_func = [x for x in nyu_data.func if "session%i" % session in x]
        session_anat = [
            x for x in nyu_data.anat_skull if "session%i" % session in x]

        for subject_id in set([os.path.basename(
                    os.path.dirname
                    (os.path.dirname(x)))
                               for x in session_func]):

            # check that subject is not condemned
            if subject_id in BAD_SUBJECTS:
                continue

            # instantiate subject_data object
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.subject_id = subject_id
            subject_data.session_id = session

            # set func
            subject_data.func = [
                x.replace(".gz", "") for x in session_func if subject_id in x]
            assert len(subject_data.func) == 1
            subject_data.func = subject_data.func[0]
            unzip_nii_gz(os.path.dirname(subject_data.func))

            # set anat
            subject_data.anat = [
                x.replace(".gz", "") for x in session_anat if subject_id in x]
            assert len(subject_data.anat) == 1
            subject_data.anat = subject_data.anat[0]
            unzip_nii_gz(os.path.dirname(subject_data.anat))

            # set subject output directory
            subject_data.output_dir = os.path.join(
                session_output_dir, subject_data.subject_id)

            yield subject_data

    # do preprocessing proper
    for session in SESSIONS:
        session_output_dir = os.path.join(OUTPUT_DIR, "session%i" % session)

        # preproprec this session for all subjects
        print ("\r\n\r\n\t\t\tPreprocessing session %i for all subjects..."
               "\r\n\r\n") % session
        nipype_preproc_spm_utils.do_subjects_preproc(
            subject_factory(session_output_dir, session),
            output_dir=session_output_dir,
            do_deleteorient=True,
            do_dartel=DARTEL,
            dataset_id="NYU Test/Retest session %i" % session,
            dataset_description=DATASET_DESCRIPTION,
            do_export_report=True,
            )

    print "\r\nDone (NYU Test/Retest)."
