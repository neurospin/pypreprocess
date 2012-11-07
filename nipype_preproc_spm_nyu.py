"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing NYU rest dataset (this is just a quick-and-dirty POC)
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
from fetch_local import fetch_nyu_data_offline

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

# set interesting subject ids
SUBJECT_IDS = ["sub18604"] # ["sub05676", "sub14864", "sub18604"]

if __name__ == '__main__':

    def subject_callback(args):
        subject_id, subject_dir, anat_image, fmri_images, session_id = args
        return do_subject_preproc(subject_id, subject_dir, anat_image,
                                  fmri_images, session_id=session_id)

    # grab local NYU directory structure
    sessions = fetch_nyu_data_offline(DATA_DIR, subject_ids=SUBJECT_IDS)

    # producer
    def preproc_factory():
        for session_id, session in sessions.iteritems():
            # pre-process data for all subjects
            for subject_id, subject in session.iteritems():
                anat_image = subject['skullstripped_anat']
                fmri_images = subject['func']
                subject_dir = os.path.join(os.path.join(DATA_DIR, session_id),
                                           subject_id)
                yield subject_id, subject_dir, anat_image, fmri_images, \
                    session_id

    results = Parallel(n_jobs=-1)(delayed(subject_callback)(args) \
                                      for args in preproc_factory())

    # generate html report (for QA)
    # report_filename = os.path.join(DATA_DIR, "_report.html")
    # report = markup.page(mode="strict_html")
    # report.p(""" pypreproc run, %s.""" % time.asctime())
    # motion_img_files = []
    # cv_tc_img_files = []
    # for subject_id, session_id, output_dirs in results:
    #     subject_dir = os.path.join(os.path.join(DATA_DIR, session_id),
    #                                subject_id)
    #     uncorrected_FMRIs = glob.glob(
    #         os.path.join(subject_dir,
    #                      "func/lfo.nii"))
    #     corrected_FMRIs = glob.glob(
    #         os.path.join(subject_dir,
    #                      "wrbet_lfo.nii"))
    #     cv_tc_plot_outfile = os.path.join(subject_dir, "cv_tc_before.png")
    #     cv_tc = plot_cv_tc(uncorrected_FMRIs, [session_id], subject_id,
    #                        plot_outfile=cv_tc_plot_outfile,
    #                        title="before preproc")
    #     cv_tc_img_files.append(cv_tc_plot_outfile)
    #     cv_tc_plot_outfile = os.path.join(subject_dir, "cv_tc_after.png")
    #     cv_tc = plot_cv_tc(corrected_FMRIs, [session_id], subject_id,
    #                        plot_outfile=cv_tc_plot_outfile,
    #                        title="after preproc")
    #     cv_tc_img_files.append(cv_tc_plot_outfile)
    #     motion_plot = plot_spm_motion_parameters(
    #         os.path.join(output_dirs["realignment"], "rp_bet_lfo.txt"),
    #         subject_id=subject_id)
    #     motion_img_files.append(motion_plot)

    # report.h1(
    #     "Plots of estimated (rigid-body) motion in original FMRI time-series")
    # report.img(src=motion_img_files)

    # report.h1(
    #     "CV (Coefficient of Variation) of corrected FMRI time-series")
    # report.img(src=cv_tc_img_files)

    # report.h1("Plots of overlays")

    # with open(report_filename, 'w') as fd:
    #     fd.write(str(report))
    #     fd.close()
