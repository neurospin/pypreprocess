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
from fetch_local import fetch_nyu_data_offline

# import spm preproc utilities
from nipype_preproc_spm_utils import do_subject_preproc, \
    do_group_preproc

# set data dir
if not 'DATA_DIR' in os.environ:
    raise IOError, "DATA_DIR is not in your environ; export it!"
DATA_DIR = os.environ['DATA_DIR']

if __name__ == '__main__':

    # grab local NYU directory structure
    sessions = fetch_nyu_data_offline(DATA_DIR)

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

    do_group_preproc(preproc_factory())
