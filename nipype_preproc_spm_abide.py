"""
:Module: nipype_preproc_spm_nyu
:Synopsis: SPM use-case for preprocessing ABIDE auditory rest dataset
:Author: dohmatob elvis dopgima

"""

# standard imports
import os
import glob
import sys

# import spm preproc utilities
import nipype_preproc_spm_utils

DATASET_DESCRIPTION = """\
<p><a href="http://fcon_1000.projects.nitrc.org/indi/abide/">ABIDE</a>\
 rest auditory dataset.</p>\
"""

# wildcard defining ABIDE directory structure
subject_id_wildcard = "*_*/*_*"

# I don't know the ABIDE sessions
SESSION_ID = "UNKNOWN_SESSION"

# DARTEL ?
DO_DARTEL = True

if __name__ == '__main__':
    # sanitize cmd-line input
    if len(sys.argv)  < 3:
        print ("\r\nUsage: source /etc/fsl/4.1/fsl.sh; python %s "
               "<path_to_ABIDE_folder> <output_dir>\r\n") % sys.argv[0]
        print ("Example:\r\nsource /etc/fsl/4.1/fsl.sh; python %s "
               "/volatile/home/aa013911/ABIDE "
               "/volatile/home/aa013911/DED/ABIDE_runs") % sys.argv[0]
        sys.exit(1)

    ABIDE_DIR = os.path.abspath(sys.argv[1])

    OUTPUT_DIR = os.path.abspath(sys.argv[2])
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if len(sys.argv) > 3:
        subject_id_wildcard = sys.argv[3]

    # glob for subject ids
    subject_ids = [os.path.basename(x)
                   for x in glob.glob(
            os.path.join(ABIDE_DIR, subject_id_wildcard))]

    subject_ids.sort()

    ignored_subject_ids = []

    # XXX the following will save you re-calculating MD5s for results
    # you know are already there

    def ignore_subject_id(subject_id):
        return os.path.exists(os.path.join(
                OUTPUT_DIR, "%s/%s/final" % (SESSION_ID, subject_id)))

    # producer subject data
    def subject_factory():
        for subject_id in subject_ids:
            # if ignore_subject_id(subject_id):
            #     continue

            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.subject_id = subject_id

            try:
                subject_data.func = glob.glob(
                    os.path.join(
                        ABIDE_DIR,
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
                        ABIDE_DIR,
                        "%s/%s/scans/anat/resources/NIfTI/files/mprage.nii" % (
                            subject_id, subject_id)))[0]
            except IndexError:
                if DO_DARTEL:
                    continue

                try:
                    subject_data.hires = glob.glob(
                        os.path.join(
                            ABIDE_DIR,
                            ("%s/%s/scans/hires/resources/NIfTI/"
                             "files/hires.nii") % (
                                subject_id, subject_id)))[0]
                except IndexError:
                    ignored_because = "no anat/hires data found"
                    print "Ignoring subject %s (%s)" % (subject_id,
                                                        ignored_because)
                    ignored_subject_ids.append((subject_id, ignored_because))
                    continue

            subject_data.output_dir = os.path.join(
                os.path.join(
                    OUTPUT_DIR, subject_data.session_id),
                subject_id)

            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(OUTPUT_DIR,
                                   "_report.html")
    nipype_preproc_spm_utils.do_group_preproc(
        subject_factory(),
        output_dir=OUTPUT_DIR,
        delete_orientation=True,
        do_dartel=DO_DARTEL,
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename)

    for subject_id, ignored_because in ignored_subject_ids:
        print "Ignored %s because %s" % (subject_id, ignored_because)
