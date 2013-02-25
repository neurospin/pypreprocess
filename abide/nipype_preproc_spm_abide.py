"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing ABIDE auditory rest dataset
:Author: dohmatob elvis dopgima

"""

"""standard imports"""
import os
import glob
import sys

sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(sys.argv[0]))))

"""import spm preproc utilities"""
import nipype_preproc_spm_utils

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
    'Phenotypic',
    'Pitt',
    'SBL',
    'SDSU',
    'Stanford',
    'Trinity',
    # 'UCLA',
    'UM',
    'USM',
    'Yale'
    ]


def preproc_abide_institute(institute_id, abide_data_dir, abide_output_dir,
                            do_realign=True,
                            do_coreg=True,     
                            do_dartel=False,
                            do_report=True,
                            do_deleteorient=True,
                            special_dirty_case=False,  # XXX voodoo option!!!
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
                   for x in glob.glob(
            os.path.join(abide_data_dir, subject_id_wildcard))]

    # sort the ids
    subject_ids.sort()

    ignored_subject_ids = []

    # producer subject data
    def subject_factory():
        for subject_id in subject_ids:
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.subject_id = subject_id

            try:
                subject_data.func = glob.glob(
                    os.path.join(
                        abide_data_dir,
                        "%s/%s/scans/rest*/resources/NIfTI/files/rest.nii" % (
                            subject_id, subject_id)))[0]
            except IndexError:
                ignored_because = "no rest data found"
                print "Ignoring subject %s (%s)" % (subject_id,
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
                             "files/hires.nii") % (
                                subject_id, subject_id)))[0]
                except IndexError:
                    ignored_because = "no anat/hires data found"
                    print "Ignoring subject %s (%s)" % (subject_id,
                                                        ignored_because)
                    ignored_subject_ids.append((subject_id, ignored_because))
                    continue

            if special_dirty_case:
                import json
                json_results_file = os.path.join(
                    "/volatile/home/edohmato/pypreproc_runs/abide",
                    institute_id,
                    subject_id,
                    "infos.json")
                if os.path.exists(json_results_file):
                    tmp = json.load(open(json_results_file, 'rb'))
                    subject_data.func = str(tmp["realigned_func"])
                    subject_data.anat = str(tmp["coregistered_anat"])

            subject_data.output_dir = os.path.join(
                os.path.join(
                    institute_output_dir, subject_id))

            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(institute_output_dir,
                                   "_report.html")
    nipype_preproc_spm_utils.do_subjects_preproc(
        subject_factory(),
        dataset_id=institute_id,
        output_dir=institute_output_dir,
        do_deleteorient=do_deleteorient,
        do_realign=do_realign,
        do_coreg=do_coreg,
        do_report=do_report,
        do_dartel=do_dartel,
        dataset_description="%s" % DATASET_DESCRIPTION.replace(
            "%s",
            institute_id),
        report_filename=report_filename,
        do_shutdown_reloaders=True,
        # do_export_report=True,
        )

    for subject_id, ignored_because in ignored_subject_ids:
        print "Ignored %s because %s" % (subject_id, ignored_because)

"""sanitize cmd-line input"""
if len(sys.argv)  < 3:
    print ("\r\nUsage: source /etc/fsl/4.1/fsl.sh; python %s "
           "<path_to_ABIDE_folder> <output_dir> [comma-separated institute"
           " ids]\r\n") % sys.argv[0]
    print ("Examples:\r\nsource /etc/fsl/4.1/fsl.sh; python %s "
           "/volatile/home/aa013911/ABIDE "
           "/volatile/home/aa013911/DED/ABIDE_runs") % sys.argv[0]
    print ("source /etc/fsl/4.1/fsl.sh; python %s "
           "/volatile/home/aa013911/ABIDE "
           "/volatile/home/aa013911/DED/ABIDE_runs Leveun,KKI,NYU"
           ) % sys.argv[0]
    sys.exit(1)

ABIDE_DIR = os.path.abspath(sys.argv[1])

OUTPUT_DIR = os.path.abspath(sys.argv[2])
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if len(sys.argv) > 3:
    INSTITUTES = sys.argv[3].split(",")

if DO_DARTEL:
    import joblib
    joblib.Parallel(n_jobs=len(INSTITUTES),
                    verbose=100)(joblib.delayed(
            preproc_abide_institute)(institute_id, ABIDE_DIR,
                                     OUTPUT_DIR,
                                     do_dartel=DO_DARTEL,
                                     do_realign=False,
                                     do_coreg=False,
                                     do_report=False,
                                     do_deleteorient=False,
                                     special_dirty_case=True)
                                 for institute_id in INSTITUTES)
else:
    for institute_id in INSTITUTES:
        preproc_abide_institute(institute_id, ABIDE_DIR, OUTPUT_DIR,
                                do_dartel=DO_DARTEL,
                                do_report=False,
                                )
