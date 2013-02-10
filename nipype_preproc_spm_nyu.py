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
import nipype_preproc_spm_utils

from external.nisl.datasets import fetch_nyu_rest, unzip_nii_gz

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
    nyu_data = fetch_nyu_rest(data_dir=sys.argv[1])

    # subject data factory
    def subject_factory():
        for j in list(xrange(len(nyu_data.subject_ids))):
            subject_data = nipype_preproc_spm_utils.SubjectData()

            subject_id = nyu_data.subject_ids[j]
            subject_data.subject_id = subject_id

            subject_data.anat = nyu_data.anat_skull[j].replace(".gz", "")
            unzip_nii_gz(os.path.dirname(subject_data.anat))

            subject_data.func = nyu_data.func[j].replace(".gz", "")
            unzip_nii_gz(os.path.dirname(subject_data.func))

            subject_data.output_dir = os.path.join(
                os.path.join(
                    OUTPUT_DIR, subject_data.session_id),
                subject_data.subject_id)

            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(OUTPUT_DIR,
                                   "_report.html")
    nipype_preproc_spm_utils.do_subjects_preproc(
        subject_factory(),
        delete_orientation=True,
        do_dartel=True,
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename)
