"""
:Module: nipype_preproc_spm_openfmri
:Synopsis: Preprocessing Openfmri
:Author: yannick schwartz, dohmatob elvis dopgima

"""

# standard imports
import os
import sys
import re
import csv
import glob
import json
import warnings

# parent dir imports
sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(sys.argv[0]))))

# import spm preproc utilities
import nipype_preproc_spm_utils

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm

# misc
from datasets_extras import unzip_nii_gz


DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""

# location of openfmri dataset on disk
DATA_ROOT_DIR = '/neurospin/tmp/havoc/openfmri_raw'

# wildcard defining directory structure
subject_id_wildcard = "sub*"

# DARTEL ?
DO_DARTEL = False

os.curdir = '..'


def apply_preproc(dataset_id, data_dir, output_dir, ignore_list=None,
                  dataset_description=None):
    """Main function for preprocessing (and analysis ?)

    Parameters
    ----------

    returns list of Bunch objects with fields anat, func, and subject_id
    for each preprocessed subject

    """
    data_dir = os.path.join(data_dir, dataset_id)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    output_dir = os.path.join(output_dir, dataset_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_description = DATASET_DESCRIPTION if \
        dataset_description is None else dataset_description

    ignore_list = [] if ignore_list is None else ignore_list

    # glob for subject ids
    subject_ids = [
        os.path.basename(x)
        for x in glob.glob(os.path.join(data_dir, subject_id_wildcard))]

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
            if subject_id in ignore_list:
                continue

            # construct subject data structure
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.session_id = session_ids
            subject_data.subject_id = subject_id
            subject_data.func = []

            # glob for bold data
            has_bad_sessions = False
            for session_id in subject_data.session_id:
                bold_dir = os.path.join(
                    data_dir,
                    "%s/BOLD/%s" % (subject_id, session_id))

                # extract .nii.gz to .nii
                unzip_nii_gz(bold_dir)

                # glob bold data for this session
                func = glob.glob(os.path.join(bold_dir, "bold.nii"))

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

            # extract .nii.gz to .ni
            unzip_nii_gz(anat_dir)

            # glob anatomical data proper
            subject_data.anat = glob.glob(
                os.path.join(
                    data_dir,
                    "%s/anatomy/highres001_brain.nii" % subject_id))[0]

            # set subject output dir (all calculations for
            # this subject go here)
            subject_data.output_dir = os.path.join(
                    output_dir,
                    subject_id)

            yield subject_data

    # do preprocessing proper
    report_filename = os.path.join(output_dir, "_report.html")
    for results in  nipype_preproc_spm_utils.do_subjects_preproc(
        subject_factory(),
        output_dir=output_dir,
        do_deleteorient=True,  # some openfmri data have garbage orientation
        do_dartel=DO_DARTEL,
        dataset_id=dataset_id,
        # do_cv_tc=False,
        dataset_description=dataset_description,
        # do_report=False,
        report_filename=report_filename,
        do_shutdown_reloaders=True,  # XXX rm this if u want to chain GLM QA
        ):
        pass

        # json_path = os.path.join(output_dir,
        #                          results['subject_id'], 'infos.json')
        # json.dump(results, open(json_path, 'wb'))


def load_preproc(dataset_id, preproc_dir, subject_ids=None):
    """Load preprocessed data from given data directory

    Parameters
    ----------
    preproc_dir: string
        directory containg data to be loaded

    subject_ids: list (optional)
        list of subject ids we are interested in (only data
        for this subjects will be loaded)

    Returns
    -------
    data: dict
       dict of lists of preprocessed bold data (4D numpy arrays),
       one per subject

    motion: dict
       dict of lists of estimated motion (2D arrays of shape n_scans x 6),
       one per subject

    """

    motion = {}
    data = {}

    preproc_dir = os.path.join(preproc_dir, dataset_id)

    # glob subject directory names
    if subject_ids is None:
        subject_dirs = sorted(glob.glob(os.path.join(preproc_dir, 'sub*')))
    else:
        subject_dirs = sorted([os.path.join(preproc_dir, subject_id)
                               for subject_id in subject_ids])
    # load data for each subject
    for subject_dir in subject_dirs:
        subject_id = os.path.split(subject_dir)[1]
        infos_path = os.path.join(subject_dir, 'infos.json')
        if os.path.exists(infos_path):
            infos = json.load(open(infos_path, 'rb'))
        else:
            continue
        if not isinstance(infos['estimated_motion'], list):
            infos['estimated_motion'] = [infos['estimated_motion']]
        if not isinstance(infos['func'], list):
            infos['func'] = [infos['func']]
        for x in infos['estimated_motion']:
            motion.setdefault(
                infos['subject_id'], []).append(np.loadtxt(os.path.join(
                    subject_dir, x.split(subject_id)[1].strip('/'))))

        data[infos['subject_id']] = [
            nb.load(os.path.join(
                subject_dir, x.split(subject_id)[1].strip('/')))
            for x in infos['func']]

    return data, motion


def load_glm_params(dataset_id, data_dir, model_id,
                    subject_ids=None,
                    hrf_model='canonical',
                    drift_model='cosine',
                    motion_params=None,
                    is_event_paradigm=True):
    """Load GLM parameters

    """

    data_dir = os.path.join(data_dir, dataset_id)
    print data_dir
    docs = []
    study_id = os.path.split(data_dir)[-1]
    tr = float(open(os.path.join(data_dir, 'scan_key.txt')).read().split()[1])

    if subject_ids is None:
        subject_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub*')))
    else:
        subject_dirs = sorted([os.path.join(data_dir, subject_id)
                               for subject_id in subject_ids])

    tasks = {}

    for subject_dir in subject_dirs:
        params = {}
        subject_id = os.path.split(subject_dir)[1]
        params['study'] = study_id
        params['subject_id'] = subject_id

        # subjects models
        model_dir = os.path.join(subject_dir, 'model', model_id)
        session_dirs = sorted(glob.glob(
            os.path.join(model_dir, 'onsets', '*')))
        # in case motion regressor are not available for all sessions
        if (motion_params is not None
            and len(motion_params[subject_id]) != len(session_dirs)):
            warnings.warn(
                'Subject [%s] Processing %d imaging session out of %d '
                '(skipping last(s)). '
                'There may be missing data, or errors in preprocessing, '
                'in particular in motion correction.' % (
                    subject_id,
                    len(motion_params[subject_id]),
                    len(session_dirs)))
            session_dirs = session_dirs[:len(motion_params[subject_id])]

        for i, session_dir in enumerate(session_dirs):
            session_id = os.path.split(session_dir)[1]
            task_id = session_id.split('_')[0]
            tasks.setdefault(task_id, []).append(i)
            img = nb.load(os.path.join(data_dir, subject_id, 'BOLD',
                                       session_id, 'bold.nii.gz'))
            n_scans = img.shape[-1]
            params.setdefault('sessions', []).append(session_id)
            params.setdefault('n_scans', []).append(n_scans)

            onsets = None
            cond_vect = None
            cond_files = sorted(glob.glob(os.path.join(session_dir, 'cond*')))
            for cond_file in cond_files:
                cond_id = int(re.findall('cond(.*).txt',
                                         os.path.split(cond_file)[1])[0])

                # load paradigm
                # replace whitespace characters by spaces before parsing
                cond = open(cond_file, 'rb').read()
                iterable = [
                    row.strip()
                    for row in re.sub('[\t\r\f\v]', ' ', cond).split('\n')]
                reader = csv.reader(iterable, delimiter=' ')
                rows = list(reader)
                if onsets is None:
                    onsets = [float(row[0]) for row in rows if row != []]
                    cond_vect = [cond_id] * len(rows)
                else:
                    onsets += [float(row[0]) for row in rows if row != []]
                    cond_vect += [cond_id] * len(rows)

            onsets = np.array(onsets)
            cond_vect = np.array(cond_vect)
            order = np.argsort(onsets)
            if is_event_paradigm:
                paradigm = EventRelatedParadigm(
                    cond_vect[order], onsets[order])
            else:
                paradigm = BlockParadigm(cond_vect[order], onsets[order])
            frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
            if motion_params is None or not subject_id in motion_params:
                design_matrix = make_dmtx(
                    frametimes, paradigm, hrf_model=hrf_model,
                    drift_model=drift_model)
            else:
                add_regs = motion_params[subject_id][i]
                add_reg_names = ['motion_%i' % r
                                 for r in range(add_regs.shape[1])]
                design_matrix = make_dmtx(
                    frametimes, paradigm, hrf_model=hrf_model,
                    drift_model=drift_model,
                    add_regs=add_regs, add_reg_names=add_reg_names)
            params.setdefault('design_matrices', []).append(
                design_matrix.matrix)

        # study model
        n_regressors = [dm.shape[-1] for dm in params['design_matrices']]

        # replace whitespace characters by spaces before parsing
        model_dir = os.path.join(data_dir, 'models', model_id)
        con = open(os.path.join(model_dir, 'task_contrasts.txt'), 'rb').read()
        iterable = [row.strip()
                    for row in re.sub('[\t\r\f\v]', ' ', con).split('\n')]
        reader = csv.reader(iterable, delimiter=' ')

        contrasts = {}
        shift = 0
        for row in reader:
            if row != []:
                for i, n_regressor in enumerate(n_regressors):
                    task_id = None
                    if row[0].startswith('task'):  # file format varies
                        shift = 1
                        task_id = row[0]

                    contrast_id = row[0 + shift]
                    if tasks.get(task_id) is None or i in tasks.get(task_id):
                        # append zeros to contrasts for the confounds
                        contrast = np.hstack(
                            [np.array(row[1 + shift:]).astype('float'),
                             np.zeros(n_regressor - len(row[1 + shift:]))])
                    else:
                        contrast = np.zeros(n_regressor)
                    contrasts.setdefault(contrast_id, []).append(contrast)

        params['contrasts'] = contrasts
        docs.append(params)

    return docs
