"""
:Module: nipype_preproc_spm_haxby
:Synopsis: SPM use-case for preprocessing HAXBY dataset
(this is just a quick-and-dirty POC)
:Author: dohmatob elvis dopgima

XXX TODO: document this according to numpy/spinx standards
XXX TODO: re-factor the code (use unittesting)
XXX TODO: more  preprocessing checks (coregistration step, etc.)
XXX TODO: over-all testing (nose ?, see with GV & BT)

"""

# standard imports
import os
import glob

# helper imports
from nisl.datasets import fetch_haxby, unzip_nii_gz

# import spm preproc utilities
from nipype_preproc_spm_utils import do_subject_preproc

# parallelism imports
from joblib import Parallel, delayed

# QA imports
from check_preprocessing import *
import markup
import time

# set data dir
if not 'DATA_DIR' in os.environ:
    raise IOError, "DATA_DIR is not in your environ; export it!"
DATA_DIR = os.environ['DATA_DIR']

if __name__ == '__main__':

    def subject_callback(args):
        # unpack args
        subject_id, subject_dir, anat_image, fmri_images = args

        # cleanup output from old runs
        for filename in glob.glob(os.path.join(subject_dir, "w*.nii")):
            os.remove(filename)

        # process this subject
        return do_subject_preproc(subject_id, subject_dir, anat_image,
                                  fmri_images)

    # fetch HAXBY dataset
    haxby_data = fetch_haxby(data_dir=DATA_DIR)

    # producer
    def preproc_factory():
        for subject_id, subject_data in haxby_data.iteritems():
            # pre-process data for all subjects
            subject_dir = subject_data["subject_dir"]
            unzip_nii_gz(subject_dir)
            anat_image = subject_data["anat"].replace(".gz", "")
            fmri_images = subject_data["bold"].replace(".gz", "")
            yield subject_id, subject_dir, anat_image, fmri_images

    # process the subjects
    results = Parallel(n_jobs=-1)(delayed(subject_callback)(args) \
                                      for args in preproc_factory())

    # generate html report (for QA)
    report_filename = os.path.join(DATA_DIR, "haxby2001/_report.html")
    report = markup.page(mode="strict_html")
    report.p(""" pypreproc run, %s.""" % time.asctime())
    report.h1(
        "Plots of estimated (rigid-body) motion in original FMRI time-series")
    for subject_id, output_dirs in results:
        motion_plot = plot_spm_motion_parameters(
            [os.path.join(output_dirs["realignment"], "rp_bet_bold.txt.png")])
        corrected_FMRIs = glob.glob(
            os.path.join(haxby_dir[subject_id]["subject_dir"],
                         "wrbet_lfo.nii"))
        print corrected_FMRIs
        cv_tc_plot = plot_cv_tc(corrected_FMRIs, session_ids=["UNKNOWN"])
        report.img(src=[motion_plot, cv_tc_plot])

    report.h1("Plots of overlays")

    with open(report_filename, 'w') as fd:
        fd.write(str(report))
        fd.close()
