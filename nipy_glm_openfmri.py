import os
import re
import json
import glob
import csv
import warnings

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm

from nipy_glm_utils import apply_glm


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

datasets_ids = {
    'ds001': 'schonberg2012decreasing',
    'ds002': 'aron2006long',
    'ds003': 'xue2011rhyme',
    'ds005': 'tom2007neural',
    'ds007': 'xue2008common',
    'ds008': 'aron2007triangulating',
    'ds011': 'foerde2006modulation',
    'ds017A': 'rizkjackson2011Aclassification',
    'ds017B': 'rizkjackson2011Bclassification',
    'ds051': 'alvarez2011cross',
    'ds052': 'poldrack2001interactive',
    'ds101': 'kelly2011simon',
    'ds102': 'kelly2008flanker',
    'ds105': 'haxby2001visual',
    'ds107': 'duncan2009consistency',
    }


def load_glm_params(data_dir, model_id,
                    subject_ids=None,
                    hrf_model='canonical',
                    drift_model='cosine',
                    motion_params=None,
                    is_event_paradigm=True):
    """Load GLM parameters

    """

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
        params['subject'] = subject_id

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


def load_preproc_data(data_dir, subject_ids=None):
    """Load preprocessed data from given data directory

    Parameters
    ----------
    data_dir: string
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

    # glob subject directory names
    if subject_ids is None:
        subject_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub*')))
    else:
        subject_dirs = sorted([os.path.join(data_dir, subject_id)
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
        if not isinstance(infos['bold'], list):
            infos['bold'] = [infos['bold']]
        for x in infos['estimated_motion']:
            motion.setdefault(
                infos['subject'], []).append(np.loadtxt(os.path.join(
                    subject_dir, x.split(subject_id)[1].strip('/'))))

        data[infos['subject']] = [
            nb.load(os.path.join(
                subject_dir, x.split(subject_id)[1].strip('/')))
            for x in infos['bold']]

    return data, motion


if __name__ == '__main__':
    # set hard coded data paths
    # preproc_root_dir = '/volatile/home/edohmato/openfmri_pypreproc_runs'
    # data_root_dir = '/neurospin/tmp/havoc/openfmri_raw'
    # out_root_dir = '/volatile/protocols/glm_open'

    data_root_dir = '/volatile/openfmri'
    preproc_root_dir = '/havoc/openfmri/preproc'
    out_root_dir = '/havoc/openfmri/glm'
    out_root_dir = '/volatile/brainpedia/protocols'

    ds_ids = sorted([
            'ds001',
            'ds002',
            'ds003',
            'ds005',
            'ds007',
            # 'ds008',
            'ds011',
            # 'ds017A',
            # 'ds017B',
            'ds051',
            'ds052',
            'ds101',
            'ds102',
            'ds105',
            'ds107'
            ])

    # ds_ids = ['ds005']

    for ds_id in ds_ids:

        # more data path business
        # ds_id = 'ds105'
        model_id = 'model001'
        ds_name = datasets[ds_id].lower().replace(' ', '_')
        data_dir = os.path.join(data_root_dir, ds_name, ds_id)
        preproc_dir = os.path.join(preproc_root_dir, ds_id)
        out_dir = os.path.join(out_root_dir, datasets_ids[ds_id])

        # load preproc data
        preproc_data, motion_params = load_preproc_data(
            preproc_dir)

        # load glm params
        glm_params = load_glm_params(data_dir, model_id,
                                     subject_ids=preproc_data.keys(),
                                     motion_params=motion_params)

        # apply glm
        apply_glm(out_dir, preproc_data, glm_params, n_jobs=-1)
