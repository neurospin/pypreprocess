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
from reporter import BASE_PREPROC_REPORT_HTML_TEMPLATE
import time

# set data dir
if not 'DATA_DIR' in os.environ:
    raise IOError, "DATA_DIR is not in your environ; export it!"
DATA_DIR = os.environ['DATA_DIR']

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
                print commands.getoutput("fslorient -deleteorient %s" \
                                             % anat_image)

                yield subject_id, subject_dir, anat_image, fmri_images, \
                    session_id

    results = Parallel(n_jobs=N_JOBS)(delayed(subject_callback)(args) \
                                      for args in preproc_factory())

    # generate html report (for QA)
    blablabla = "Generating QA report for %d subjects .." % len(results)
    dadada = "+" * len(blablabla)
    print "\r\n%s\r\n%s\r\n%s\r\n" % (dadada, blablabla, dadada)

    tmpl = BASE_PREPROC_REPORT_HTML_TEMPLATE
    report_filename = "nyu_preproc_report.html"
    plots_gallery = list()

    for subject_id, session_id, output in results:
        full_plot = output['plots']['segmentation']
        title = 'subject: %s' % subject_id
        summary_plot = output['plots']['segmentation_summary']
        redirect_url = output["report"]
        plots_gallery.append((full_plot, title, summary_plot, redirect_url))

    report  = tmpl.substitute(now=time.ctime(), plots_gallery=plots_gallery)
    print report
    with open(report_filename, 'w') as fd:
        fd.write(str(report))
        fd.close()

    print "\r\nDone."
