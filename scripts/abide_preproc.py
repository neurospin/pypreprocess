"""
:Module: nipype_preproc_spm_abide
:Synopsis: SPM use-case for preprocessing ABIDE rest dataset
:Author: dohmatob elvis dopgima

"""

"""standard imports"""
import os
import glob
import sys
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData
                                                   )

# brief description of ABIDE
DATASET_DESCRIPTION = """\
<p><a href="http://fcon_1000.projects.nitrc.org/indi/abide/">Institute %s, \
ABIDE</a> rest auditory dataset.</p>\
"""

"""DARTEL ?"""
DO_DARTEL = True

"""institutes we're insterested in"""
INSTITUTES = [
    'CMU',
    'Caltech',
    'KKI',
    'Leuven',
    'MaxMun',
    'NYU',
    'OHSU',
    'Olin',
    'Pitt',
    'SBL',
    'SDSU',
    'Stanford',
    'Trinity',
    'UCLA',
    'UM',
    'USM',
    'Yale']


def preproc_abide_institute(institute_id, abide_data_dir, abide_output_dir,
                            do_dartel=True,
                            do_report=True,
                            n_jobs=-1,
                            ):
    """Preprocesses a given ABIDE institute

    """

    # set institute output dir
    institute_output_dir = os.path.join(abide_output_dir, institute_id)
    if not os.path.exists(institute_output_dir):
        os.makedirs(institute_output_dir)

    # set subject id wildcard for globbing institute subjects
    subject_id_wildcard = "%s_*/%s_*" % (institute_id, institute_id)

    # glob for subject ids
    subject_ids = [os.path.basename(x)
                   for x in glob.glob(os.path.join(abide_data_dir,
                                                   subject_id_wildcard))]

    # sort the ids
    subject_ids.sort()

    ignored_subject_ids = []

    # producer subject data
    def subject_factory():
        for subject_id in subject_ids:
            subject_data = SubjectData()
            subject_data.subject_id = subject_id

            try:
                subject_data.func = glob.glob(
                    os.path.join(
                        abide_data_dir,
                        "%s/%s/scans/rest*/resources/NIfTI/files/rest.nii" % (
                            subject_id, subject_id)))[0]
            except IndexError:
                ignored_because = "no rest data found"
                print("Ignoring subject %s (%s)" % (subject_id,)
                                                    ignored_because)
                ignored_subject_ids.append((subject_id, ignored_because))
                continue

            try:
                subject_data.anat = glob.glob(
                    os.path.join(
                        abide_data_dir,
                        "%s/%s/scans/anat/resources/NIfTI/files/mprage.nii" % (
                            subject_id, subject_id)))[0]
            except IndexError:
                if do_dartel:
                    # can't do DARTEL in under such conditions
                    continue

                try:
                    subject_data.hires = glob.glob(
                        os.path.join(
                            abide_data_dir,
                            ("%s/%s/scans/hires/resources/NIfTI/"
                             "files/hires.nii") % (subject_id, subject_id)))[0]
                except IndexError:
                    ignored_because = "no anat/hires data found"
                    print("Ignoring subject %s (%s)" % (subject_id,)
                                                        ignored_because)
                    ignored_subject_ids.append((subject_id, ignored_because))
                    continue

            subject_data.output_dir = os.path.join(
                os.path.join(
                    institute_output_dir, subject_id))

            # yield data for this subject
            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(institute_output_dir,
                                   "_report.html")
    do_subjects_preproc(
        subject_factory(),
        dataset_id=institute_id,
        output_dir=institute_output_dir,
        do_report=do_report,
        do_dartel=do_dartel,
        dataset_description="%s" % DATASET_DESCRIPTION.replace(
            "%s",
            institute_id),
        report_filename=report_filename,
        do_shutdown_reloaders=True,)

    for subject_id, ignored_because in ignored_subject_ids:
        print("Ignored %s because %s" % (subject_id, ignored_because))

"""sanitize cmd-line input"""
if len(sys.argv) < 3:
    print("\r\nUsage: source /etc/fsl/4.1/fsl.sh; python %s "
          "<path_to_ABIDE_folder> <output_dir> [comma-separated institute"
          " ids]\r\n" % sys.argv[0])
    print("Examples:\r\nsource /etc/fsl/4.1/fsl.sh; python %s "
          "/volatile/home/aa013911/ABIDE "
          "/volatile/home/aa013911/DED/ABIDE_runs" % sys.argv[0])
    print("source /etc/fsl/4.1/fsl.sh; python %s "
          "/volatile/home/aa013911/ABIDE "
          "/volatile/home/aa013911/DED/ABIDE_runs Leveun,KKI,NYU"
          % sys.argv[0])
    sys.exit(1)

ABIDE_DIR = os.path.abspath(sys.argv[1])

OUTPUT_DIR = os.path.abspath(sys.argv[2])
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if len(sys.argv) > 3:
    INSTITUTES = sys.argv[3].split(",")

if DO_DARTEL:
    import joblib
    joblib.Parallel(n_jobs=1, verbose=100)(
        joblib.delayed(preproc_abide_institute)(
            institute_id,
            ABIDE_DIR,
            OUTPUT_DIR,
            do_dartel=True,
            # do_report=False,
            )
        for institute_id in INSTITUTES)
else:
    for institute_id in INSTITUTES:
        preproc_abide_institute(institute_id, ABIDE_DIR, OUTPUT_DIR,
                                do_dartel=False,
                                do_report=False,
                                )
