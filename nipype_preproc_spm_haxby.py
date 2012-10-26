"""
:Module: nipype_preproc_spm_haxby
:Synopsis: SPM use-case for preprocessing HAXBY rest dataset
(this is just a quick-and-dirty POC)
:Author: dohmatob elvis dopgima

XXX TODO: document this according to numpy/spinx standards
XXX TODO: re-factor the code (use unittesting)
XXX TODO: visualization
XXX TODO: preprocessing checks
XXX TODO: over-all testing (nose ?, see with GV & BT)

"""

# standard imports
import os

# helper imports
from fetch_local import fetch_haxby_data_offline

# import spm preproc utilities
from nipype_preproc_spm_utils import do_subject_preproc

# parallelism imports
from joblib import Parallel, delayed

# import visualization tools
from check_preprocessing import *

# set data dir
DATA_DIR = os.getcwd()
if 'DATA_DIR' in os.environ:
    DATA_DIR = os.environ['DATA_DIR']
    assert os.path.exists(DATA_DIR), \
    "DATA_DIR: %s, doesn't exist" % DATA_DIR

# set interesting subject ids
SUBJECT_IDS = ["subj1", "subj2", "subj3", "subj4"]

if __name__ == '__main__':

    def subject_callback(args):
        # unpack args
        subject_id, subject_dir, anat_image, fmri_images = args

        # cleanup output from old runs
        for filename in glob.glob(os.path.join(subject_dir, "w*.nii")):
            os.remove(filename)

        # process this subject
        do_subject_preproc(subject_id, subject_dir, anat_image, fmri_images)

    # grab local HAXBY directory structure
    sessions = fetch_haxby_data_offline(DATA_DIR,
                                        subject_ids=SUBJECT_IDS)

    # producer
    def producer():
        for session_id, session in sessions.iteritems():
            # pre-process data for all subjects
            for subject_id, subject in session.iteritems():
                anat_image = subject['anat']
                fmri_images = subject['func']
                subject_dir = os.path.join(os.path.join(DATA_DIR,
                                                        session_id),
                                           subject_id)

                yield subject_id, subject_dir, anat_image, fmri_images

    # process the subjects
    Parallel(n_jobs=-1)(delayed(subject_callback)(args) for args in producer())

    # run checks
    import pylab as pl
    for session_id, session in sessions.iteritems():
        # pre-process data for all subjects
        for subject_id, subject in session.iteritems():
            anat_image = subject['anat']
            fmri_images = subject['func']
            subject_dir = os.path.join(os.path.join(DATA_DIR,
                                                    session_id),
                                       subject_id)
            plot_cv_tc(glob.glob(os.path.join(subject_dir, "w*.nii")),
                       [session_id],
                       subject_id)

    pl.show()
