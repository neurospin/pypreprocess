"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing NYU rest dataset
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
import commands

# helper imports
from fetch_local import fetch_nyu_data_offline

# import spm preproc utilities
from nipype_preproc_spm_utils import do_subject_preproc

# parallelism imports
from joblib import Parallel, delayed
from multiprocessing import cpu_count

# QA imports
from check_preprocessing import *
import markup
from report_utils import *
import time

# set data dir
if not 'DATA_DIR' in os.environ:
    raise IOError, "DATA_DIR is not in your environ; export it!"
DATA_DIR = os.environ['DATA_DIR']

# set interesting subject ids
SUBJECT_IDS = ["sub05676", "sub08889", "sub14864", "sub08224"]

# set job count
N_JOBS = -1
if 'N_JOBS' in os.environ:
    N_JOBS = int(os.environ['N_JOBS'])

if __name__ == '__main__':

    def subject_callback(args):
        subject_id, subject_dir, anat_image, fmri_images, session_id = args
        return do_subject_preproc(subject_id, subject_dir, anat_image,
                                  fmri_images, session_id=session_id)

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
                commands.getoutput("fslorient -deleteorient %s" % anat_image)

                yield subject_id, subject_dir, anat_image, fmri_images, \
                    session_id

    results = Parallel(n_jobs=N_JOBS)(delayed(subject_callback)(args) \
                                      for args in preproc_factory())

    # generate html report (for QA)
    report_filename = os.path.join(DATA_DIR, "_report.html")
    report = markup.page(mode="strict_html")
    report.h1(""" pypreproc run, %s.""" % time.asctime())
    report.p("See reports for each stage below.")
    plots = dict()
    plots["cv_tc"] = list()
    plots["realignment"] = list()
    plots["coregistration"] = list()
    plots["segmentation"] = list()

    for subject_id, session_id, output in results:
        report.a("Report for subject %s " % subject_id, class_="internal",
                 href=output["report"])
        plots["cv_tc"].append(output["plots"]["cv_tc"])
        plots["realignment"].append(output["plots"]["realignment_parameters"])
        plots["coregistration"].append(output["plots"]["coregistration"])
        plots["segmentation"].append(output["plots"]["segmentation"])

    report.h2("Grouped reports")

    report.a("Coefficient of Variation report", class_='internal',
             href="cv_tc_report.html")
    report.br()
    cv_tc_report_filename = os.path.join(DATA_DIR,
                                                "cv_tc_report.html")
    cv_tc_report = markup.page(mode='loose_html')
    cv_tc_report.h1(
        "Coefficient of Variation report for %i subjects" % len(results))
    sidebyside(cv_tc_report,
               plots["cv_tc"][:len(plots["cv_tc"]) / 2],
               plots["cv_tc"][len(plots["cv_tc"]) / 2:])
    with open(cv_tc_report_filename, 'w') as fd:
        fd.write(str(cv_tc_report))
        fd.close()

    report.a("Realignment report", class_='internal',
             href="realignment_report.html")
    report.br()
    realignment_report_filename = os.path.join(DATA_DIR,
                                                "realignment_report.html")
    realignment_report = markup.page(mode='loose_html')
    realignment_report.h1("Realignment report for %i subjects" % len(results))
    sidebyside(realignment_report,
               plots["realignment"][:len(plots["realignment"]) / 2],
               plots["realignment"][len(plots["realignment"]) / 2:])
    with open(realignment_report_filename, 'w') as fd:
        fd.write(str(realignment_report))
        fd.close()

    report.a("Coregistration report", class_='internal',
             href="coregistration_report.html")
    report.br()
    coregistration_report_filename = os.path.join(DATA_DIR,
                                                "coregistration_report.html")
    coregistration_report = markup.page(mode='loose_html')
    coregistration_report.h1(
        "Coregistration report for %i subjects" % len(results))
    sidebyside(coregistration_report,
               plots["coregistration"][:len(plots["coregistration"]) / 2],
               plots["coregistration"][len(plots["coregistration"]) / 2:])
    with open(coregistration_report_filename, 'w') as fd:
        fd.write(str(coregistration_report))
        fd.close()

    report.a("Segmentation report", class_='internal',
             href="segmentation_report.html")
    report.br()
    segmentation_report_filename = os.path.join(DATA_DIR,
                                                "segmentation_report.html")
    segmentation_report = markup.page(mode='loose_html')
    segmentation_report.h1(
        "Segmentation report for %i subjects" % len(results))
    sidebyside(segmentation_report,
               plots["segmentation"][:len(plots["segmentation"]) / 2],
               plots["segmentation"][len(plots["segmentation"]) / 2:])
    with open(segmentation_report_filename, 'w') as fd:
        fd.write(str(segmentation_report))
        fd.close()

    with open(report_filename, 'w') as fd:
        fd.write(str(report))
        fd.close()
