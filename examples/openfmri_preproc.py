"""
:Module: nipype_preproc_spm_openfmri
:Synopsis: Preprocessing Openfmri
:Author: yannick schwartz, dohmatob elvis dopgima

"""

# standard imports
import os
import glob
import traceback

# import spm preproc utilities
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData)
from pypreprocess.datasets import fetch_openfmri

DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""

# DARTEL ?
DO_DARTEL = False


def preproc_dataset(data_dir, output_dir, dataset_id=None,
                    ignore_subjects=None, n_jobs=-1):
    """Main function for preprocessing (and analysis ?)

    Parameters
    ----------

    returns list of Bunch objects with fields anat, func, and subject_id
    for each preprocessed subject

    """
    ignore_subjects = [] if ignore_subjects is None else ignore_subjects

    # glob for subjects and their imaging sessions identifiers
    subjects = [os.path.basename(x)
                for x in glob.glob(os.path.join(data_dir, 'sub???'))]

    sessions = set()
    for subject_id in subjects:
        subject_dir = os.path.join(data_dir, subject_id)
        for session_dir in glob.glob(os.path.join(
                subject_dir, 'model', '*', 'onsets', '*')):
            sessions.add(os.path.split(session_dir)[1])

    sessions = sorted(sessions)
    subjects = sorted(subjects)

    # producer subject data
    def subject_factory():
        for subject_id in subjects:
            if subject_id in ignore_subjects:
                continue

            # construct subject data structure
            subject_data = SubjectData()
            subject_data.session_id = sessions
            subject_data.subject_id = subject_id
            subject_data.func = []

            # glob for bold data
            has_bad_sessions = False
            for session_id in subject_data.session_id:
                bold_dir = os.path.join(
                    data_dir, subject_id, 'BOLD', session_id)

                # glob bold data for this session
                func = glob.glob(os.path.join(bold_dir, "bold.nii.gz"))

                # check that this session is OK (has bold data, etc.)
                if not func:
                    has_bad_sessions = True
                    break

                subject_data.func.append(func[0])

            # exclude subject if necessary
            if has_bad_sessions:
                continue

            # anatomical data
            subject_data.anat = os.path.join(
                data_dir, subject_id, 'anatomy', 'highres001_brain.nii.gz')
            if not os.path.exists(subject_data.anat):
                subject_data.anat = os.path.join(
                    data_dir, subject_id, 'anatomy', 'highres001.nii.gz')

            # subject output_dir
            subject_data.output_dir = os.path.join(output_dir, subject_id)
            yield subject_data

    return do_subjects_preproc(
        subject_factory(),
        n_jobs=n_jobs,
        dataset_id=dataset_id,
        output_dir=output_dir,
        deleteorient=True,
        dartel=DO_DARTEL,
        dataset_description=DATASET_DESCRIPTION,
        )

if __name__ == '__main__':
    import sys
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = '%prog [input_dir] [output_dir]'
    parser.description = (
        '`input_dir` is the path to an existing '
        'OpenfMRI dataset or where to download it. '
        'The directory name must match a valid OpenfMRI dataset id, '
        'and therefore look like /path/to/dir/{dataset_id}.')

    parser.add_option(
        '-n', '--n-jobs', dest='n_jobs', type='int',
        default=os.environ.get('N_JOBS', '1'),
        help='Number of parallel jobs.')

    options, args = parser.parse_args(sys.argv)
    input_dir, output_dir = args[1:]
    input_dir = input_dir.rstrip('/')
    output_dir = output_dir.rstrip('/')
    data_dir, dataset_id = os.path.split(input_dir)
    if not dataset_id.startswith('ds'):
        parser.error("The directory does not seem to be an OpenfMRI dataset.")

    fetch_openfmri(data_dir, dataset_id)

    preproc_dataset(data_dir=input_dir,
         output_dir=output_dir,
         n_jobs=options.n_jobs)

    print "\r\nAll output written to %s" % output_dir
