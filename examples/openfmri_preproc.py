"""
:Module: nipype_preproc_spm_openfmri
:Synopsis: Preprocessing Openfmri
:Author: yannick schwartz, dohmatob elvis dopgima

"""

# standard imports
import os
import glob
import warnings

# import spm preproc utilities
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData)
from pypreprocess.datasets import fetch_openfmri

DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""

# DARTEL ?
DO_DARTEL = False


def preproc_dataset(data_dir, output_dir,
                    ignore_subjects=None, restrict_subjects=None,
                    n_jobs=-1):
    """Main function for preprocessing a dataset with the OpenfMRI layout.

    Parameters
    ----------
    data_dir: str
        Path of input directory. If does not exist and finishes
        by a valid OpenfMRI dataset id, it will be downloaded.
    output_dir: str
        Path of output directory.
    ignore_subjects: list or None
        List of subject identifiers not to process.
    restrict_subjects: list or None
        List of subject identifiers to process.
    n_jobs: int
        Number of parallel processes.

    Warning
    -------
    Subjects may be excluded if some data is missing.

    Returns list of Bunch objects with fields anat, func, and subject_id
    for each preprocessed subject
    """
    parent_dir, dataset_id = os.path.split(data_dir)

    if not os.path.exists(data_dir):
        fetch_openfmri(parent_dir, dataset_id)

    ignore_subjects = [] if ignore_subjects is None else ignore_subjects

    # glob for subjects and their imaging sessions identifiers
    if restrict_subjects is None:
        subjects = [os.path.basename(x)
                    for x in glob.glob(os.path.join(data_dir, 'sub???'))]
    else:
        subjects = restrict_subjects

    subjects = sorted(subjects)

    # producer subject data
    def subject_factory():
        for subject_id in subjects:
            if subject_id in ignore_subjects:
                continue

            sessions = set()
            subject_dir = os.path.join(data_dir, subject_id)
            for session_dir in glob.glob(os.path.join(
                    subject_dir, 'BOLD', '*')):
                sessions.add(os.path.split(session_dir)[1])
            sessions = sorted(sessions)
            # construct subject data structure
            subject_data = SubjectData()
            subject_data.session_id = sessions
            subject_data.subject_id = subject_id
            subject_data.func = []

            # glob for BOLD data
            has_bad_sessions = False
            for session_id in subject_data.session_id:
                bold_dir = os.path.join(
                    data_dir, subject_id, 'BOLD', session_id)

                # glob BOLD data for this session
                func = glob.glob(os.path.join(bold_dir, "bold.nii.gz"))
                # check that this session is OK (has BOLD data, etc.)
                if not func:
                    warnings.warn(
                        'Subject %s is missing data for session %s.' % (
                        subject_id, session_id))
                    has_bad_sessions = True
                    break

                subject_data.func.append(func[0])

            # exclude subject if necessary
            if has_bad_sessions:
                warnings.warn('Excluding subject %s' % subject_id)
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
        deleteorient=False,
        dartel=DO_DARTEL,
        dataset_description=DATASET_DESCRIPTION,
        # caching=False,
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
        '-s', '--subjects', dest='subjects',
        help='Process a single subject matching the given id.')

    parser.add_option(
        '-n', '--n-jobs', dest='n_jobs', type='int',
        default=os.environ.get('N_JOBS', '1'),
        help='Number of parallel jobs.')

    options, args = parser.parse_args(sys.argv)
    input_dir, output_dir = args[1:]
    input_dir = input_dir.rstrip('/')
    output_dir = output_dir.rstrip('/')
    if not dataset_id.startswith('ds') and not os.path.exists(input_dir):
        parser.error("The directory does not exist and "
                     "does not seem to be an OpenfMRI dataset.")

    if options.subjects is not None and os.path.exists(options.subjects):
        with open(options.subjects, 'rb') as f:
            restrict = f.read().split()
    else:
        restrict = None if options.subjects is None else [options.subjects]

    preproc_dataset(data_dir=input_dir,
         output_dir=output_dir,
         restrict_subjects=restrict,
         n_jobs=options.n_jobs)

    print "\r\nAll output written to %s" % output_dir
