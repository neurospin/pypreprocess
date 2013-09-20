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
                                                   SubjectData
                                                   )

# misc
from pypreprocess.datasets_extras import unzip_nii_gz

DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""

# location of openfmri dataset on disk
DATA_ROOT_DIR = '/neurospin/tmp/openfmri'

# wildcard defining directory structure
subject_id_wildcard = "sub*"

# DARTEL ?
DO_DARTEL = False

# openfmri datasets we are interested in
datasets = {
    'ds001': 'Balloon Analog Risk-taking Task',
    'ds002': 'Classification learning',
    'ds003': 'Rhyme judgment',
    'ds005': 'Mixed-gambles task',
    'ds007': 'Stop-signal task with spoken & manual responses',
    'ds008': 'Stop-signal task with unselective and selective stopping',
    'ds011': 'Classification learning and tone-counting',
    'ds017A': ('Classification learning and '
               'stop-signal (1 year test-retest)'),
    'ds017B': ('Classification learning and '
               'stop-signal (1 year test-retest)'),
    'ds051': 'Cross-language repetition priming',
    'ds052': 'Classification learning and reversal',
    'ds101': 'Simon task dataset',
    'ds102': 'Flanker task (event-related)',
    'ds105': 'Visual object recognition',
    'ds107': 'Word and object processing',
    }

datasets_exclusions = {}


def main(data_dir, output_dir, exclusions=None, dataset_id=None,
         n_jobs=-1):
    """Main function for preprocessing (and analysis ?)

    Parameters
    ----------

    returns list of Bunch objects with fields anat, func, and subject_id
    for each preprocessed subject

    """
    exclusions = [] if exclusions is None else exclusions

    # glob for subject ids
    subject_ids = [
        os.path.basename(x)
        for x in glob.glob(os.path.join(data_dir,
                                        subject_id_wildcard))]

    model_dirs = glob.glob(os.path.join(
        data_dir, subject_ids[0], 'model', '*'))

    session_ids = [
        os.path.basename(x)
        for x in glob.glob(os.path.join(model_dirs[0], 'onsets', '*'))]

    session_ids.sort()
    subject_ids.sort()

    # producer subject data
    def subject_factory():
        for subject_id in subject_ids:
            if subject_id in exclusions:
                continue

            # construct subject data structure
            subject_data = SubjectData()
            subject_data.session_id = session_ids
            subject_data.subject_id = subject_id
            subject_data.func = []

            # glob for bold data
            has_bad_sessions = False
            for session_id in subject_data.session_id:
                bold_dir = os.path.join(
                    data_dir,
                    "%s/BOLD/%s" % (subject_id, session_id))

                # # extract .nii.gz to .nii
                # unzip_nii_gz(bold_dir)

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

            # glob for anatomical data
            anat_dir = os.path.join(
                data_dir,
                "%s/anatomy" % subject_id)

            # # extract .nii.gz to .ni
            # unzip_nii_gz(anat_dir)

            # glob anatomical data proper
            subject_data.anat = glob.glob(
                os.path.join(
                    data_dir,
                    "%s/anatomy/highres001_brain.nii.gz" % subject_id))[0]

            # set subject output dir (all calculations for
            # this subject go here)
            subject_data.output_dir = os.path.join(
                    output_dir,
                    subject_id)

            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(output_dir,
                                   "_report.html")
    return do_subjects_preproc(
        subject_factory(),
        n_jobs=n_jobs,
        dataset_id=dataset_id,
        output_dir=output_dir,
        do_deleteorient=True,  # some openfmri data have garbage orientation
        do_dartel=DO_DARTEL,
        # do_cv_tc=False,
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename,
        do_shutdown_reloaders=True
        )

if __name__ == '__main__':
    n_jobs = int(os.environ['N_JOBS']) if 'N_JOBS' in os.environ else -1

    # where output will be spat; replace as necessary
    output_root_dir = '/volatile/home/edohmato/openfmri_pypreproc_runs'

    # dataset ids we're interested in
    ds_ids = sorted([
            'ds001',
            'ds002',
            'ds003',
            'ds005',
            'ds007',
            'ds008',
            'ds011',
            'ds017A',
            'ds017B',
            'ds051',
            'ds052',
            'ds101',
            'ds102',
            'ds105',
            'ds107'
            ])

    # /!\ Don't try to 'parallelize' this loop!!!
    for ds_id in ds_ids:
        try:
            data_dir = os.path.join(DATA_ROOT_DIR, ds_id)
            output_dir = os.path.join(output_root_dir, ds_id)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            results = main(data_dir, output_dir,
                           datasets_exclusions.get(ds_id),
                           dataset_id=ds_id,
                           n_jobs=n_jobs)
        except:
            print traceback.format_exc()
            pass

    print "\r\nAll output written to %s" % output_root_dir
