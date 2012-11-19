"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing NYU rest dataset
(this is just a quick-and-dirty POC)
:Author: dohmatob elvis dopgima

XXX TODO: document this according to numpy/spinx standards
XXX TODO: re-factor the code (use unittesting)
XXX TODO: over-all testing (nose ?, see with GV & BT)

"""

# standard imports
import os
import commands

# helper imports
import fetch_local

# import spm preproc utilities
import nipype_preproc_spm_utils

# set data dir
if not 'DATA_DIR' in os.environ:
    raise IOError, "DATA_DIR is not in your environ; export it!"
DATA_DIR = os.environ['DATA_DIR']

DATASET_DESCRIPTION = """\
The NYU CSC TestRetest resource includes EPI-images of 25 participants\
 gathered during rest as well as anonymized anatomical images of the \
same participants.

The resting-state fMRI images were collected on several occasions:
1. the first resting-state scan in a scan session
2. 5-11 months after the first resting-state scan
3. about 30 (< 45) minutes after 2.

<a href="http://www.nitrc.org/projects/nyu_trt/"> NYU TestRest</a>
"""

if __name__ == '__main__':

    # grab local NYU directory structure
    sessions = fetch_local.fetch_nyu_data_offline(
        DATA_DIR)

    # producer
    def preproc_factory():
        for session_id, session in sessions.iteritems():
            # pre-process data for all subjects
            for subject_id, subject in session.iteritems():
                anat_image = subject['skullstripped_anat']
                fmri_images = subject['func']
                subject_dir = os.path.join(os.path.join(DATA_DIR, session_id),
                                           subject_id)

                # anats for some subjects have shitty orientation (LR, AP, SI)
                # meta-headers (and this leads to awefully skrewed-up coreg!)
                # strip them off, and let SPM figure out the right orientaion
                print commands.getoutput("fslorient -deleteorient %s" \
                                             % anat_image)

                yield subject_id, subject_dir, anat_image, fmri_images, \
                    session_id

    # do preprocessing proper
    nipype_preproc_spm_utils.do_group_preproc(
        preproc_factory(),
        dataset_description=DATASET_DESCRIPTION,
        report_filename=os.path.join(DATA_DIR,
                                     "nyu_preproc_report.html"))
