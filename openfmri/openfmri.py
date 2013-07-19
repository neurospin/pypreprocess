import os
import sys
import re
import glob
import json
import shutil
import tempfile
import urllib2
import multiprocessing

from optparse import OptionParser

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.glm import FMRILinearModel


# import externals
fpath = os.path.abspath(sys.argv[0])
module_dir = os.path.join(os.path.sep, *fpath.split(os.path.sep)[:-2])

sys.path.append(module_dir)
sys.path.append(os.path.join(module_dir, 'external'))

from nilearn.datasets import _fetch_dataset
from datasets_extras import unzip_nii_gz
import nipype_preproc_spm_utils

from datasets import dataset_names, dataset_files, map_id
from datasets import dataset_ignore_list

# wildcard defining directory structure
subject_id_wildcard = "sub*"

# DARTEL ?
DO_DARTEL = False
os.curdir = '..'

DATASET_DESCRIPTION = """\
<p><a href="https://openfmri.org/data-sets">openfmri.org datasets</a>.</p>
"""


def normalize_name(name):
    return re.sub("[^0-9a-zA-Z\-]+", '_', name).lower().strip('_')

# ----------------------------------------------------------------------------
# download openfmri data
# ----------------------------------------------------------------------------


def fetch_openfmri(dataset_id, data_dir, redownload=False):
    """ Downloads and extract datasets from www.openfmri.org

        Parameters
        ----------
        accession_number: str
            Dataset identifier, short version of https://openfmri.org/data-sets
        data_dir: str
            Destination directory.
        redownload: boolean
            Set to True to force redownload of already available data.
            Defaults to False.

        Datasets
        --------
        {accession_number}: {dataset name}
        ds001: Balloon Analog Risk-taking Task
        ds002: Classification learning
        ds003: Rhyme judgment
        ds005: Mixed-gambles task
        ds007: Stop-signal task with spoken & manual responses
        ds008: Stop-signal task with unselective and selective stopping
        ds011: Classification learning and tone-counting
        ds017: Classification learning and stop-signal (1 year test-retest)
        ds051: Cross-language repetition priming
        ds052: Classification learning and reversal
        ds101: Simon task dataset
        ds102: Flanker task (event-related)
        ds105: Visual object recognition
        ds107: Word and object processing

        Returns
        -------
        ds_path: str
            Path of the dataset.
    """
    ds_url = 'https://openfmri.org/system/files/%s.tgz'
    ds_name = normalize_name(dataset_names[dataset_id])
    ds_urls = [ds_url % name for name in dataset_files[dataset_id]]
    dl_path = os.path.join(data_dir, ds_name)
    ds_path = os.path.join(data_dir, dataset_id)

    if not os.path.exists(ds_path) or redownload:
        if os.path.exists(ds_path):
            shutil.rmtree(ds_path)
        _fetch_dataset(ds_name, ds_urls, data_dir, verbose=1)

        shutil.move(os.path.join(dl_path, dataset_id), ds_path)
        shutil.rmtree(dl_path)

    return ds_path


# ----------------------------------------------------------------------------
# parse openfmri layout
# ----------------------------------------------------------------------------

def get_study_tr(study_dir):
    return float(
        open(os.path.join(study_dir, 'scan_key.txt')).read().split()[1])


def get_sessions_task(subject_dir):
    sessions = os.path.join(subject_dir, 'BOLD', '*')
    return [os.path.split(session)[1].split('_')[0]
            for session in sorted(glob.glob(sessions))]


def get_tasks_runs(subject_dir):
    # get runs identifiers, checks that bold runs and onset runs match.
    # also checks that onset folders contain at least one condition file.
    sessions = os.path.join(subject_dir, 'BOLD', '*')
    bold_sessions = [os.path.split(session)[1]
                     for session in sorted(glob.glob(sessions))]
    sessions = os.path.join(subject_dir, 'model', 'model001', 'onsets', '*')
    onset_sessions = [os.path.split(session)[1]
                     for session in sorted(glob.glob(sessions))
                     if os.path.exists(os.path.join(session, 'cond001.txt'))]
    return sorted(set(bold_sessions).intersection(onset_sessions))


def get_task_conditions(subject_dir, task_id):
    task_dir = glob.glob(os.path.join(subject_dir, 'model', 'model001',
                            'onsets', '%s_*' % task_id))[0]

    conditions = glob.glob(os.path.join(task_dir, '*.txt'))
    return sorted([os.path.split(cond)[1] for cond in conditions])


def get_motion(subject_dir):
    sessions = os.path.join(subject_dir, 'BOLD', '*')

    motion = []
    for session_dir in sorted(glob.glob(sessions)):
        motion_s = open(os.path.join(session_dir, 'motion.txt')).read()
        motion_s = np.array([l.split() for l in motion_s.split('\n')][:-1])
        motion.append(np.array(motion_s).astype('float'))
    return motion


def get_bold_images(subject_dir):
    sessions = get_tasks_runs(subject_dir)
    # sessions = os.path.join(subject_dir, 'BOLD', '*')

    images = []
    for session_id in sessions:
        session_dir = os.path.join(subject_dir, 'BOLD', session_id)
        img = nb.load(os.path.join(session_dir, 'normalized_bold.nii.gz'))

        images.append(img)

    n_scans = [img.shape[-1] for img in images]

    return images, n_scans


def get_task_contrasts(study_dir, subject_dir, model_id):
    contrasts_path = os.path.join(
        study_dir, 'models', model_id, 'task_contrasts.txt')

    do_all_tasks = False
    task_contrasts = {}
    for line in open(contrasts_path, 'rb').read().split('\n')[:-1]:
        line = line.split()
        task_id = line[0]
        contrast_id = line[1]
        con_val = np.array(line[2:]).astype('float')

        if 'task' not in task_id:       # alternative format
            do_all_tasks = True
            task_id = 'task001'
            contrast_id = line[0]
            con_val = np.array(line[1:]).astype('float')

        task_contrasts.setdefault(task_id, {}).setdefault(contrast_id, con_val)

    if do_all_tasks:
        tasks = set([
            run.split('_')[0] for run in get_sessions_task(subject_dir)])
        for task_id in tasks:
            task_contrasts[task_id] = task_contrasts['task001']

    ordered = {}
    #  do a get cond length with number of con in sub/model/onset/condi.txt
    for task_id in sorted(task_contrasts.keys()):
        for contrast_id in task_contrasts[task_id]:
            for session_task_id in get_sessions_task(subject_dir):
                if session_task_id == task_id:
                    con_val = task_contrasts[task_id][contrast_id]
                else:
                    n_conds = len(get_task_conditions(subject_dir, task_id))
                    con_val = np.array([0] * n_conds)
                ordered.setdefault(contrast_id, []).append(con_val)
    return ordered


def get_events(subject_dir):
    sessions = get_tasks_runs(subject_dir)
    # sessions = os.path.join(subject_dir, 'model', 'model001', 'onsets', '*')

    events = []

    for session_id in sessions:
        session_dir = os.path.join(subject_dir,
                                   'model', 'model001', 'onsets', session_id)
        conditions = glob.glob(os.path.join(session_dir, 'cond*.txt'))
        onsets = []
        cond_id = []
        for i, path in enumerate(sorted(conditions)):
            # cond_onsets = open(path, 'rb').read().split('\n')
            # cond_onsets = [l.split() for l in cond_onsets[:-1]]
            # cond_onsets = np.array(cond_onsets).astype('float')
            cond_onsets = np.loadtxt(path)
            onsets.append(cond_onsets)
            cond_id.append([i] * cond_onsets.shape[0])

        events.append((np.vstack(onsets), np.concatenate(cond_id)))

    return events


def make_design_matrices(events, n_scans, tr, hrf_model='canonical',
                         drift_model='cosine', motion=None):

    design_matrices = []
    n_sessions = len(n_scans)

    for i in range(n_sessions):

        onsets = events[i][0][:, 0]
        duration = events[i][0][:, 1]
        amplitude = events[i][0][:, 2]
        cond_id = events[i][1]
        order = np.argsort(onsets)

        # make a block or event paradigm depending on stimulus duration
        if duration.sum() == 0:
            paradigm = EventRelatedParadigm(cond_id[order],
                                            onsets[order],
                                            amplitude[order])
        else:
            paradigm = BlockParadigm(cond_id[order], onsets[order],
                                     duration[order], amplitude[order])

        frametimes = np.linspace(0, (n_scans[i] - 1) * tr, n_scans[i])

        if motion is not None:
            add_regs = np.array(motion[i]).astype('float')
            add_reg_names = ['motion_%i' % r
                             for r in range(add_regs.shape[1])]

            design_matrix = make_dmtx(
                frametimes, paradigm, hrf_model=hrf_model,
                drift_model=drift_model,
                add_regs=add_regs, add_reg_names=add_reg_names)
        else:
            design_matrix = make_dmtx(
                frametimes, paradigm, hrf_model=hrf_model,
                drift_model=drift_model)

        design_matrices.append(design_matrix.matrix)

    return design_matrices


# ----------------------------------------------------------------------------
# dump openfmri layout from spm
# ----------------------------------------------------------------------------

def write_new_model(study_dir, model_id, contrasts):
    models_dir = os.path.join(study_dir, 'models')

    if not os.path.exists(os.path.join(models_dir, model_id)):
        os.makedirs(os.path.join(models_dir, model_id))

    cond_model001 = os.path.join(models_dir, 'model001', 'condition_key.txt')
    cond_model002 = os.path.join(models_dir, model_id, 'condition_key.txt')

    shutil.copyfile(cond_model001, cond_model002)

    contrasts_path = os.path.join(models_dir, model_id, 'task_contrasts.txt')

    with open(contrasts_path, 'wb') as f:
        for contrast in contrasts:
            task_id, contrast_id = contrast.split('__')
            con_val = contrasts[contrast]
            con_val = ' '.join(np.array(con_val).astype('|S32'))
            f.write('%s %s %s\n' % (task_id, contrast_id, con_val))


def spm_to_openfmri(out_dir, preproc_docs, intra_docs, metadata=None,
                    n_jobs=-1, verbose=1):
    metadata = _check_metadata(metadata, preproc_docs[0], intra_docs[0])

    _openfmri_metadata(os.path.join(out_dir, metadata['study_id']), metadata)

    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    docs = zip(preproc_docs, intra_docs)

    if n_jobs == 1:
        for i, (preproc_doc, intra_doc) in enumerate(docs):
            _openfmri_preproc(out_dir, preproc_doc, metadata, verbose)
            _openfmri_intra(out_dir, intra_doc, metadata, verbose)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        for i, (preproc_doc, intra_doc) in enumerate(docs):
            pool.apply_async(_openfmri_preproc,
                             args=(out_dir, preproc_doc, metadata, verbose))
            pool.apply_async(_openfmri_intra,
                             args=(out_dir, intra_doc, metadata, verbose))
        pool.close()
        pool.join()


def _check_metadata(metadata, preproc_doc, intra_doc):
    if metadata is None:
        metadata = {}

    if not 'run_key' in metadata:
        metadata['run_key'] = ['task%03i run%03i' % (1, i + 1)
                           for i in range(len(preproc_doc['n_scans']))]

    if not 'condition_key' in metadata:
        metadata['condition_key'] = {}
        for run_key, conditions in zip(metadata['run_key'],
                                       intra_doc['condition_key']):
            metadata['condition_key'][run_key] = conditions

    if not 'scan_key' in metadata:
        metadata['scan_key'] = {}
        metadata['scan_key']['TR'] = intra_doc['tr']

    if 'study_id' in intra_doc:
        metadata['study_id'] = intra_doc['study_id']
    else:
        metadata['study_id'] = ''

    return metadata


def _openfmri_preproc(out_dir, doc, metadata=None, verbose=1):
    """
        Parameters
        ----------
        metadata: dict
            - run_key: naming the sessions

        Examples
        --------
        {'run_key': ['task001 run001', 'task001 run002',
                     'task002 run001', 'task002 run002']}

    """
    if 'study_id' in doc:
        study_dir = os.path.join(out_dir, doc['study_id'])
    else:
        study_dir = out_dir

    if verbose > 0:
        print '%s@%s: dumping preproc' % (doc['subject_id'], doc['study_id'])

    subject_dir = os.path.join(study_dir, doc['subject_id'])
    anatomy_dir = os.path.join(subject_dir, 'anatomy')

    if not os.path.exists(anatomy_dir):
        os.makedirs(anatomy_dir)

    anatomy = doc['preproc']['anatomy']
    wm_anatomy = doc['final']['anatomy']

    anatomy = nb.load(anatomy)
    wm_anatomy = nb.load(wm_anatomy)

    nb.save(anatomy, os.path.join(anatomy_dir, 'highres001.nii.gz'))
    nb.save(wm_anatomy, os.path.join(anatomy_dir,
                                     'normalized_highres001.nii.gz'))

    bold_dir = os.path.join(subject_dir, 'BOLD')

    for session, run_key in zip(
            doc['slice_timing']['bold'], metadata['run_key']):

        bold = nb.concat_images(session)
        session_dir = os.path.join(bold_dir, run_key.replace(' ', '_'))
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        nb.save(bold, os.path.join(session_dir, 'bold.nii.gz'))

    for session, motion, run_key in zip(doc['final']['bold'],
                                        doc['realign']['motion'],
                                        metadata['run_key']):

        bold = nb.concat_images(session)
        session_dir = os.path.join(bold_dir, run_key.replace(' ', '_'))
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        nb.save(bold, os.path.join(session_dir, 'normalized_bold.nii.gz'))
        shutil.copyfile(motion, os.path.join(session_dir, 'motion.txt'))


def _openfmri_intra(out_dir, doc, metadata=None, verbose=1):
    """
        Parameters
        ----------
        metadata: dict
            - condition_key
              https://openfmri.org/content/metadata-condition-key

        Examples
        --------
        {'condition_key': {'task001 cond001': 'task',
                           'task001 cond002': 'parametric gain'}}
    """
    if 'study_id' in doc:
        study_dir = os.path.join(out_dir, doc['study_id'])
    else:
        study_dir = out_dir

    if verbose > 0:
        print '%s@%s: dumping stats intra' % (doc['subject_id'],
                                              doc['study_id'])

    subject_dir = os.path.join(study_dir, doc['subject_id'])

    model_dir = os.path.join(study_dir, 'models', 'model001')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # conditions specification
    conditions_spec = []
    for key, val in sorted(metadata['condition_key'].iteritems()):
        for i, name in enumerate(val):
            conditions_spec.append(
                '%s cond%03i %s\n' % (key.split(' ')[0], i + 1, name))

    with open(os.path.join(model_dir, 'condition_key.txt'), 'wb') as f:
                f.write(''.join(sorted(set(conditions_spec))))

    # contrasts specification
    contrasts_spec = []
    for key, val in doc['task_contrasts'].iteritems():
        if 'task_contrasts' in metadata:
            key = doc['task_contrasts'][key]

        for i, session_contrast in enumerate(val):
            task_id = metadata['run_key'][i].split(' ')[0]
            # check not null and 1d
            if (np.abs(session_contrast).sum() > 0
                and len(np.array(session_contrast).shape) == 1):
                con = ' '.join(np.array(session_contrast).astype('|S32'))
                contrasts_spec.append('%s %s %s\n' % (task_id, key, con))

    with open(os.path.join(model_dir, 'task_contrasts.txt'), 'wb') as f:
        f.write(''.join(sorted(set(contrasts_spec))))

    # dump onsets
    model_dir = os.path.join(subject_dir, 'model', 'model001')
    onsets_dir = os.path.join(model_dir, 'onsets')

    for onsets, run_key in zip(doc['onsets'], metadata['run_key']):
        run_dir = os.path.join(onsets_dir, run_key.replace(' ', '_'))
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        for condition_id, values in onsets.iteritems():
            cond = os.path.join(run_dir, '%s.txt' % condition_id)
            with open(cond, 'wb') as f:
                for timepoint in values:
                    f.write('%s %s %s\n' % timepoint)

    # analyses
    for dtype in ['c_maps', 't_maps']:
        data_dir = os.path.join(model_dir, dtype)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if isinstance(doc[dtype], dict):
            for contrast_id in doc[dtype].keys():
                fname = normalize_name(contrast_id)
                img = nb.load(doc[dtype][contrast_id])
                nb.save(img, os.path.join(data_dir, '%s.nii.gz' % fname))

    # general data for analysis
    img = nb.load(doc['mask'])
    nb.save(img, os.path.join(model_dir, 'mask.nii.gz'))
    json.dump(doc, open(os.path.join(model_dir, 'SPM.json'), 'wb'))


def _openfmri_metadata(out_dir, metadata):
    """ General dataset information

        Parameters
        ----------
        metadata: dict
            - task_key -- https://openfmri.org/content/metadata-task-key
            - scan_key -- https://openfmri.org/content/metadata-scan-key

        Examples
        --------
        {'task_key': {'task001': 'stop signal with manual response',
                      'task002': 'stop signal with letter naming'}}
        {'scan_key': {'TR': 2.0}
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # naming the tasks
    if 'task_key' in metadata:
        with open(os.path.join(out_dir, 'task_key.txt'), 'wb') as f:
            for key, val in sorted(metadata['task_key'].iteritems()):
                f.write('%s %s\n' % (key, val))

    # scanning info
    if 'scan_key' in metadata:
        with open(os.path.join(out_dir, 'scan_key.txt'), 'wb') as f:
            for key, val in sorted(metadata['scan_key'].iteritems()):
                f.write('%s %s\n' % (key, val))

    # extra info, for example subject_id mapping etc...
    if 'extras' in metadata:
        meta_dir = os.path.join(out_dir, 'metadata')
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir)
        for key, val in metadata['extras'].iteritems():
            with open(os.path.join(meta_dir, '%s.txt' % key), 'wb') as f:
                for k, v in sorted(val.iteritems()):
                    f.write('%s %s\n' % (k, v))


# ----------------------------------------------------------------------------
# GLM on openfmri layout
# ----------------------------------------------------------------------------


def first_level_glm(study_dir, subjects_id, model_id,
                     hrf_model='canonical', drift_model='cosine',
                     glm_model='ar1', mask='compute', n_jobs=-1, verbose=1):

    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    print study_dir, subjects_id

    if n_jobs == 1:
        for subject_id in subjects_id:
            _first_level_glm(study_dir, subject_id, model_id,
                        hrf_model=hrf_model,
                        drift_model=drift_model,
                        glm_model=glm_model, mask=mask, verbose=verbose)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        for subject_id in subjects_id:
            pool.apply_async(
                _first_level_glm,
                args=(study_dir, subject_id, model_id),
                kwds={'hrf_model': hrf_model,
                      'drift_model': drift_model,
                    'glm_model': glm_model, 'mask': mask, 'verbose': verbose})

        pool.close()
        pool.join()


def _first_level_glm(study_dir, subject_id, model_id,
                     hrf_model='canonical', drift_model='cosine',
                     glm_model='ar1', mask='compute', verbose=1):

    study_id = os.path.split(study_dir)[1]
    subject_dir = os.path.join(study_dir, subject_id)

    if verbose > 0:
        print '%s@%s: first level glm' % (subject_id, study_id)

    tr = get_study_tr(study_dir)
    images, n_scans = get_bold_images(subject_dir)
    motion = get_motion(subject_dir)
    contrasts = get_task_contrasts(study_dir, subject_dir, model_id)
    events = get_events(subject_dir)

    design_matrices = make_design_matrices(events, n_scans, tr,
                                           hrf_model, drift_model, motion)

    glm = FMRILinearModel(images, design_matrices, mask=mask)
    glm.fit(do_scaling=True, model=glm_model)

    for contrast_id in contrasts:

        con_val = []
        for session_con, session_dm in zip(contrasts[contrast_id],
                                           design_matrices):
            con = np.zeros(session_dm.shape[1])
            con[:len(session_con)] = session_con
            con_val.append(con)

        z_map, t_map, c_map, var_map = glm.contrast(
            con_val,
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True,)

        model_dir = os.path.join(subject_dir, 'model',  model_id)

        for dtype, img in zip(['z', 't', 'c', 'var'],
                              [z_map, t_map, c_map, var_map]):

            map_dir = os.path.join(model_dir, '%s_maps' % dtype)

            if not os.path.exists(map_dir):
                os.makedirs(map_dir)

            path = os.path.join(
                map_dir, '%s.nii.gz' % normalize_name(contrast_id))
            nb.save(img, path)

    nb.save(glm.mask, os.path.join(model_dir, 'mask.nii.gz'))

# ----------------------------------------------------------------------------
# launcher stuff
# ----------------------------------------------------------------------------


def get_dataset_description(dataset_id):
    full_id = map_id[dataset_id]
    html = urllib2.urlopen('https://openfmri.org/dataset/%s' % full_id).read()
    return html.split(('<div class="field-item even" '
                       'property="content:encoded"><p>')
           )[1].split(('</p>\n</div></div></div><div class="field '
                       'field-name-field-mixedformattasksconditions'))[0]


def get_options(args):
    parser = OptionParser()
    parser.add_option(
        '-t', '--dataset', dest='dataset_id',
        help='The openfmri dataset id.\nSee https://openfmri.org/data-sets')
    parser.add_option('-d', '--dataset-dir', dest='dataset_dir',
        help='Parent path for the dataset.')
    parser.add_option('-m', '--model', dest='model_id', default='model001',
                      help='The model to be used from the GLM.')
    parser.add_option(
        '-p', '--preprocessing', dest='preproc_dir',
        default=tempfile.gettempdir(),
        help='Parent path for preprocessing.')
    parser.add_option(
        '-f', '--force-dowload', dest='force_download',
        action='store_true', default=False,
        help='Force redownload the dataset.')
    parser.add_option(
        '-v', '--verbose', dest='verbose',
        type='int', default=1,
        help='Verbosity level.')
    parser.add_option(
        '-a', '--skip-preprocessing', dest='skip_preprocessing',
        action='store_true', default=False,
        help='Force preprocessing skipping.')

    parser.add_option(
        '-u', '--subject_id', dest='subject_id',
        help='If defined, only this subject is processed.')

    options, args = parser.parse_args(args)

    dataset_dir = options.dataset_dir
    dataset_id = options.dataset_id

    if dataset_id is None:
        parser.error("The dataset id is mandatory.")
    if dataset_dir is None:
        parser.error("The data directory is mandatory.")

    return options


def setup_dataset(options):
    dataset_id = options.dataset_id
    dataset_dir = options.dataset_dir
    preproc_dir = options.preproc_dir

    if not os.path.exists(preproc_dir):
        os.makedirs(preproc_dir)

    if options.verbose > 0:
        print 'Fetching data...'
    dataset_dir = fetch_openfmri(dataset_id, dataset_dir,
                                 redownload=options.force_download)

    if options.verbose > 0:
        print 'Copying models...'
    # update and/or create models
    model001_dir = os.path.join(dataset_dir, 'models', 'model001')
    for task_contrasts_path in glob.glob(os.path.join('models', '*')):
        ds_id, model_id, f = os.path.split(task_contrasts_path)[1].split('__')

        if ds_id == dataset_id:
            model_dir = os.path.join(dataset_dir, 'models', model_id)
            if not os.path.exists(model_dir):
                shutil.copytree(model001_dir, model_dir)

            shutil.copyfile(task_contrasts_path, os.path.join(model_dir, f))


def process_dataset(argv):
    options = get_options(argv)

    dataset_id = options.dataset_id
    model_id = options.model_id
    preproc_dir = options.preproc_dir
    dataset_dir = options.dataset_dir

    ignore_list = dataset_ignore_list[dataset_id]
    description = get_dataset_description(dataset_id)

    study_dir = os.path.join(dataset_dir, dataset_id)

    if options.subject_id is not None:
        ignore_list = [os.path.split(p)[1]
                       for p in glob.glob(os.path.join(study_dir, 'sub*'))
                       if os.path.split(p)[1] != options.subject_id]

    subjects_id = [os.path.split(p)[1]
                   for p in glob.glob(os.path.join(study_dir, 'sub*'))
                   if os.path.split(p)[1] not in ignore_list]

    setup_dataset(options)
    if options.verbose > 0:
        print 'Preprocessing data...'
    if not options.skip_preprocessing:
        dataset_preprocessing(dataset_id, dataset_dir, preproc_dir,
                              ignore_list, description)

    first_level_glm(study_dir, subjects_id, model_id, n_jobs=1,
                    verbose=options.verbose)


def dataset_preprocessing(dataset_id, data_dir, output_dir, ignore_list=None,
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
    subjects_id = [
        os.path.basename(x)
        for x in glob.glob(os.path.join(data_dir, subject_id_wildcard))]
    subjects_id.sort()

    sessions_id = {}

    # producer subject data
    def subject_factory():
        for subject_id in subjects_id:
            if subject_id in ignore_list:
                continue

            sessions = get_tasks_runs(os.path.join(data_dir, subject_id))
            sessions_id[subject_id] = sessions

            # construct subject data structure
            subject_data = nipype_preproc_spm_utils.SubjectData()
            subject_data.session_id = sessions
            subject_data.subject_id = subject_id
            subject_data.func = []

            assert sessions != []

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
        n_jobs=1,
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

        subject_id = results['subject_id']

        # dump results in openfmri layout
        if not isinstance(results['estimated_motion'], list):
            results['estimated_motion'] = [results['estimated_motion']]
        if not isinstance(results['func'], list):
            results['func'] = [results['func']]

        img = nb.load(results['anat'])
        nb.save(img, os.path.join(
            data_dir, subject_id, 'anatomy',
            'normalized_highres001.nii.gz'))

        for session_id, motion, func in zip(sessions_id[subject_id],
                                            results['estimated_motion'],
                                            results['func']):

            # estimated motion
            shutil.copyfile(motion, os.path.join(
                data_dir, subject_id, 'BOLD', session_id, 'motion.txt'))

            # preprocessed bold
            img = nb.load(func)
            nb.save(img, os.path.join(
                data_dir, subject_id, 'BOLD',
                session_id, 'normalized_bold.nii.gz'))
