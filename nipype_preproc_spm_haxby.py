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
from nisl.datasets import fetch_haxby, unzip_nii_gz

# import spm preproc utilities
from nipype_preproc_spm_utils import do_subject_preproc
# parallelism imports
from joblib import Parallel, delayed

# import visualization tools
from check_preprocessing import *
import pylab as pl

# set data dir
if not 'DATA_DIR' in os.environ:
    raise RuntimeError, "DATA_DIR is not in your environ; export it!"
DATA_DIR = os.environ['DATA_DIR']

if __name__ == '__main__':

    def subject_callback(args):
        # unpack args
        subject_id, subject_dir, anat_image, fmri_images = args

        # cleanup output from old runs
        for filename in glob.glob(os.path.join(subject_dir, "w*.nii")):
            os.remove(filename)

        # process this subject
        do_subject_preproc(subject_id, subject_dir, anat_image, fmri_images)

    # fetch HAXBY dataset
    haxby_data = fetch_haxby(data_dir=DATA_DIR)

    # producer
    def producer():
        for subject_id, subject_data in haxby_data.iteritems():
            # pre-process data for all subjects
            subject_dir = subject_data["subject_dir"]
            unzip_nii_gz(subject_dir)
            anat_image = subject_data["anat"].replace(".gz", "")
            fmri_images = subject_data["bold"].replace(".gz", "")
            yield subject_id, subject_dir, anat_image, fmri_images

    # process the subjects
    Parallel(n_jobs=-1)(delayed(subject_callback)(args) for args in producer())

    # do some 'visual' QA
    for subject_id, subject_data in haxby_data.iteritems():
        subject_dir = subject_data["subject_dir"]

        # XXX TODO: Merged plots (using subplotting) of CV and motion params
        # plot CV map
        plot_cv_tc(glob.glob(os.path.join(subject_dir, "w*.nii")),
                   ["haxby2001"],
                   subject_id)

        # plot motion parameters
        parameter_files = glob.glob("%s/realign/rp_bold.txt" % subject_id)
        assert len(parameter_files) == 1
        plot_spm_motion_parameters(parameter_files, [subject_id])

    pl.show()
