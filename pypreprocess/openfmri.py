# standard imports
import os
import glob
import warnings

# import spm preproc utilities
from .nipype_preproc_spm_utils import (do_subjects_preproc, SubjectData)
from .datasets import fetch_openfmri

DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""


def preproc_dataset(data_dir, output_dir,
                    ignore_subjects=None, restrict_subjects=None,
                    delete_orient=False, dartel=False,
                    n_jobs=-1):
    """Main function for preprocessing a dataset with the OpenfMRI layout.

    Parameters
    ----------
    data_dir: str
        Path of input directory. If does not exist and finishes
        by a valid OpenfMRI dataset id, it will be downloaded,
        i.e., /path/to/dir/{dataset_id}.
    output_dir: str
        Path of output directory.
    ignore_subjects: list or None
        List of subject identifiers not to process.
    restrict_subjects: list or None
        List of subject identifiers to process.
    delete_orient: bool
        Delete orientation information in nifti files.
    dartel: bool
        Use dartel.
    n_jobs: int
        Number of parallel jobs.

    Examples
    --------
    preproc_dataset('/tmp/ds105', '/tmp/ds105_preproc ',
                    ignore_subjects=['sub002', 'sub003'],
                    delete_orient=True,
                    n_jobs=3)

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
                data_dir, subject_id, 'anatomy', 'highres001.nii.gz')
            # pypreprocess is setup to work with non-skull stripped brain and
            # is likely to crash otherwise.
            if not os.path.exists(subject_data.anat):
                subject_data.anat = os.path.join(
                    data_dir, subject_id, 'anatomy', 'highres001_brain.nii.gz')

            # subject output_dir
            subject_data.output_dir = os.path.join(output_dir, subject_id)
            yield subject_data

    return do_subjects_preproc(
        subject_factory(),
        n_jobs=n_jobs,
        dataset_id=dataset_id,
        output_dir=output_dir,
        deleteorient=delete_orient,
        dartel=dartel,
        dataset_description=DATASET_DESCRIPTION,
        # caching=False,
        )
