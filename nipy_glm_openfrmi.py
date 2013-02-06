import os
import re
import json
import glob
import csv

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


def load_glm_params(data_dir, model_id,
                    hrf_model='canonical',
                    drift_model='cosine',
                    motion_params=None,
                    is_event_paradigm=True):

    docs = []
    study_id = os.path.split(data_dir)[-1]
    tr = float(open(os.path.join(data_dir, 'scan_key.txt')).read().split()[1])

    subject_dirs = sorted(glob.glob(os.path.join(data_dir, 'sub*')))
    for subject_dir in subject_dirs:
        params = {}
        subject_id = os.path.split(subject_dir)[1]
        params['study'] = study_id
        params['subject'] = subject_id

        # subjects models
        model_dir = os.path.join(subject_dir, 'model', model_id)
        session_dirs = sorted(glob.glob(
            os.path.join(model_dir, 'onsets', '*')))
        for i, session_dir in enumerate(session_dirs):
            session_id = os.path.split(session_dir)[1]
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

        docs.append(params)

    # study model
    n_regressors = [dm.shape[-1] for dm in params['design_matrices']]
    model_dir = os.path.join(data_dir, 'models', model_id)

    # replace whitespace characters by spaces before parsing
    con = open(os.path.join(model_dir, 'task_contrasts.txt'), 'rb').read()
    iterable = [row.strip()
                for row in re.sub('[\t\r\f\v]', ' ', con).split('\n')]
    reader = csv.reader(iterable, delimiter=' ')
    # reader = csv.reader(contrasts_file, delimiter=' ')

    contrasts = {}
    shift = 0
    for row in reader:
        if row != []:
            for n_regressor in n_regressors:
                if row[0].startswith('task'):  # file format varies
                    shift = 1
                contrast_id = row[0 + shift]
                # append zeros to contrasts for the confounds
                contrast = np.hstack(
                    [np.array(row[1 + shift:]).astype('float'),
                     np.zeros(n_regressor - len(row[1 + shift:]))])

                contrasts.setdefault(contrast_id, []).append(contrast)

    for doc in docs:
        doc['contrasts'] = contrasts

    return docs


def load_preproc(data_dir):
    motion = {}
    data = {}

    for subject_dir in glob.glob(os.path.join(data_dir, 'sub*')):
        infos = json.load(open(os.path.join(subject_dir, 'infos.json'), 'rb'))
        subject_id = infos['subject']
        if not isinstance(infos['estimated_motion'], list):
            infos['estimated_motion'] = [infos['estimated_motion']]
        if not isinstance(infos['bold'], list):
            infos['bold'] = [infos['bold']]
        for fname in infos['estimated_motion']:
            motion.setdefault(subject_id, []).append(np.loadtxt(fname))

        data[infos['subject_id']] = [nb.load(x) for x in infos['bold']]

    return data, motion


if __name__ == '__main__':
    preproc_root_dir = '/volatile/home/edohmato/openfmri_pypreproc_runs'
    data_root_dir = '/neurospin/tmp/havoc/openfmri_raw'
    out_root_dir = '/volatile/protocols/glm_open'

    ds_id = 'ds011'
    model_id = 'model001'

    ds_name = datasets[ds_id].lower().replace(' ', '_')

    data_dir = os.path.join(data_root_dir, ds_name, ds_id)
    preproc_dir = os.path.join(preproc_root_dir, ds_id)
    out_dir = os.path.join(out_root_dir, ds_name, ds_id)

    preproc_data, motion_params = load_preproc(preproc_dir)
    glm_params = load_glm_params(data_dir, model_id,
                                 motion_params=motion_params)

    apply_glm(out_dir, preproc_data, glm_params, n_jobs=-1)
