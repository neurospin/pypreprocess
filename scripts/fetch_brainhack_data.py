"""
:Synopsis: br41nh4ck d4t4 gr4bB3r!
:Author: d0hm4t06 3Lv15 <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import os
import sys
import re
import glob
import commands
from joblib import delayed, Parallel
import nibabel

# global constants
DATA_DIR = os.path.join(os.getcwd(), "BrainHack_data")
BOLD_URL_PATTERN = ("ftp://ftp.mrc-cbu.cam.ac.uk/personal/"
                    "rik.henson/wakemandg_hensonrn/Sub%02i/BOLD/Run_%02i/")
ANAT_URL_PATTERN = ("ftp://ftp.mrc-cbu.cam.ac.uk/personal/rik.henson"
                    "/wakemandg_hensonrn/Sub%02i/T1")
DONE_URLS = []


def _file_exists(url, output_dir):
    """
    Checks whether to-be-downloaded stuff already exists locally

    """

    if url in DONE_URLS:
        return True

    output_filename = os.path.join(output_dir, os.path.basename(url))
    if not os.path.exists(output_filename):
        return False

    for ext in ['.txt', '.mat']:
        if output_filename.endswith(ext):
            if os.path.isfile(output_filename):
                # print "Skipping existing file: %s" % output_filename
                DONE_URLS.append(url)
                return True

    if output_filename.endswith(".nii"):
        try:
            nibabel.load(output_filename)
            nibabel.concat_images([output_filename])
            # print "Skipping existing file: %s" % output_filename
            DONE_URLS.append(url)
            return True
        except Exception, e:
            print "nibabel.load(...) error:", e
            print
            print "Corrupt image %s; redownloading" % output_filename
            print commands.getoutput("rm -f %s*" % output_filename)
            return False


def _download_url(url, output_dir):
    """
    Actually download stuff

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, os.path.basename(url))

    old_cwd = os.getcwd()
    os.chdir(output_dir)

    wget_cmd = "wget %s" % url
    print "Executing: %s ..." % wget_cmd
    print commands.getoutput(wget_cmd)
    os.chdir(old_cwd)

    return output_filename


def _url_factory():
    """
    Generate tasks (urls) for workers (crawlers)

    """

    for subject_id in range(1, 2):
        # anat nifti URLs
        anat_url = ANAT_URL_PATTERN % (subject_id + 1)
        anat_dir = os.path.join(DATA_DIR, "Sub%02i/T1" % (
                subject_id + 1))
        for item in ['mprage.nii',
                     'mri_fids.txt',
                     'headpoints.txt'
                     ]:
            item_url = os.path.join(anat_url, item)

            if _file_exists(item_url, anat_dir):
                continue

            yield item_url, anat_dir

        for run_id in xrange(9):
            # fMRI nifti URLs
            bold_url = BOLD_URL_PATTERN % (subject_id + 1, run_id + 1)
            # req = urllib2.urlopen(bold_url)
            run_dir = os.path.join(DATA_DIR, "Sub%02i/Run_%02i" % (
                    subject_id + 1, run_id + 1))
            # dump = req.read()
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            tmp = os.path.join(run_dir, "index.html*")
            commands.getoutput("rm -f %s" % tmp)
            old_cwd = os.getcwd()
            os.chdir(run_dir)
            commands.getoutput("wget %s" % bold_url)
            index_file = os.path.join(run_dir, "index.html")
            if not os.path.isfile(index_file):
                continue
            dump = open(index_file, 'r').read()
            os.chdir(old_cwd)
            for item in re.finditer('<a href="(?P<url>.+?\.nii)">', dump):
                item_url = item.group("url")

                if _file_exists(item_url, run_dir):
                    continue

                yield item_url, run_dir

            # paradigm info URLs
            for ext in ["txt"]:
                item_url = os.path.join(os.path.dirname(bold_url), "Trials",
                                         "run_%02i_spmdef.%s" % (
                        run_id + 1, ext)
                                         )

                if _file_exists(item_url, run_dir):
                    continue

                yield item_url, run_dir


def get_subject_data_from_disk(subject_id):
    """
    Grab data for a given subject

    """

    return dict(
        subject_id=subject_id,

        # run IDs
        session_id=["Run_%02i" % (run_id + 1) for run_id in xrange(9)],

        # functional data fMR*.nii
        func=[sorted(glob.glob(os.path.join(DATA_DIR, subject_id,
                                                    "Run_%02i/fMR*.nii" % (
                            run_id + 1)))) for run_id in xrange(9)],

        # anat
        anat=os.path.join(DATA_DIR, subject_id, "T1/mprage.nii"),

        # timing files (run_spmdef.txt)
        timing=[os.path.join(DATA_DIR, subject_id,
                             "Run_%02i/run_%02i_spmdef.txt" % (
                    run_id + 1, run_id + 1)) for run_id in xrange(9)])


def download_all():
    """
    Download all the data from the internet

    """

    Parallel(n_jobs=int(os.environ.get('N_JOBS', -1)), verbose=100)(
        delayed(_download_url)(url, output_dir)
        for url, output_dir in _url_factory())


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_DIR = os.path.abspath(sys.argv[1])

    download_all()
