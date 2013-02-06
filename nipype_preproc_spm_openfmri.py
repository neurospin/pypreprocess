"""
:Module: nipype_preproc_spm_openfmri_ds107
:Synopsis: Preprocessing Openfmri ds107
:Author: dohmatob elvis dopgima

"""

# standard imports
import os
import glob
import sys
import json

# import spm preproc utilities
import nipype_preproc_spm_utils

# misc
from external.nisl.datasets import unzip_nii_gz

DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""

# wildcard defining directory structure
subject_id_wildcard = "sub*"

# DARTEL ?
DO_DARTEL = False


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

datasets_exclusions = {
    'ds017A': ['sub003'],
    'ds017B': ['sub003'],
    'ds007': ['sub009', 'sub018'],
    'ds051': ['sub006'],
}


def main(DATA_DIR, OUTPUT_DIR, exclusions=None):
    """
    returns list of Bunch objects with fields anat, func, and subject_id
    for each preprocessed subject

    """
    exclusions = [] if exclusions is None else exclusions

    # glob for subject ids
    subject_ids = [
        os.path.basename(x)
        for x in glob.glob(os.path.join(DATA_DIR, subject_id_wildcard))]

    model_dirs = glob.glob(os.path.join(
        DATA_DIR, subject_ids[0], 'model', '*'))

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
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.session_id = session_ids
            subject_data.subject_id = subject_id
            subject_data.func = []

            # orientation meta-data for sub013 is garbage
            if subject_id in ['sub013'] and not DO_DARTEL:
                subject_data.bad_orientation = True

            # glob for bold data
            for session_id in subject_data.session_id:
                bold_dir = os.path.join(
                    DATA_DIR,
                    "%s/BOLD/%s" % (subject_id, session_id))

                # extract .nii.gz to .nii
                unzip_nii_gz(bold_dir)

                # glob bold data proper
                func = glob.glob(
                    os.path.join(
                        DATA_DIR,
                        "%s/BOLD/%s/bold.nii" % (
                            subject_id, session_id)))[0]
                subject_data.func.append(func)

            # glob for anatomical data
            anat_dir = os.path.join(
                DATA_DIR,
                "%s/anatomy" % subject_id)

            # extract .nii.gz to .ni
            unzip_nii_gz(anat_dir)

            # glob anatomical data proper
            subject_data.anat = glob.glob(
                os.path.join(
                    DATA_DIR,
                    "%s/anatomy/highres001_brain.nii" % subject_id))[0]

            # set subject output dir (all calculations for
            # this subject go here)
            subject_data.output_dir = os.path.join(
                    OUTPUT_DIR,
                    subject_id)

            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(OUTPUT_DIR,
                                   "_report.html")
    return nipype_preproc_spm_utils.do_group_preproc(
        subject_factory(),
        output_dir=OUTPUT_DIR,
        # delete_orientation=True,
        do_dartel=DO_DARTEL,
        do_cv_tc=False,
        do_report=False,
        do_export_report=True,
        dataset_description=DATASET_DESCRIPTION,
        report_filename=report_filename
        )

if __name__ == '__main__':
    data_root_dir = '/volatile/openfmri'
    out_root_dir = '/havoc/openfmri/preproc'

    ds_ids = [
        # 'ds001',
        # 'ds002',
        # 'ds003',
        # 'ds005',
        # 'ds007',
        # 'ds008',
        # 'ds011',
        # 'ds017A',
        # 'ds017B',
        # 'ds051',
        # 'ds052',
        'ds101',
        # 'ds102',
        # 'ds105',
        # 'ds107'
        ]

    for ds_id in ds_ids:
        ds_name = datasets[ds_id].lower().replace(' ', '_')
        data_dir = os.path.join(data_root_dir, ds_name, ds_id)
        out_dir = os.path.join(out_root_dir, ds_id)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        results = main(data_dir, out_dir, datasets_exclusions.get(ds_id))

        for res in results:
            infos = {}
            infos['anat'] = res[1]['anat']
            infos['estimated_motion'] = res[1]['estimated_motion']
            infos['bold'] = res[1]['func']
            infos['subject'] = res[1]['subject_id']
            path = os.path.join(out_dir, infos['subject'], 'infos.json')
            json.dump(infos, open(path, 'wb'))
