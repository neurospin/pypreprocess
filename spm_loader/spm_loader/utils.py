import os
import sys
import re
import time
import json
import joblib
import multiprocessing

import nibabel as nb
import numpy as np

from nipy.modalities.fmri.glm import FMRILinearModel

# pypreprocess imports
PYPREPROCESS_DIR = os.path.dirname(
    os.path.dirname(os.path.split(os.path.abspath(__file__))[0]))
sys.path.append(PYPREPROCESS_DIR)
import reporting.glm_reporter as glm_reporter
import reporting.base_reporter as base_reporter
from nipype_preproc_spm_utils import do_subjects_preproc, SubjectData


def fix_docs(docs, fix=None, fields=None):
    if fields is None:
        fields = ['t_maps', 'c_maps', 'c_maps_smoothed', 'contrasts']
    if fix is None or fix == {}:
        return docs
    fixed_docs = []

    for doc in docs:
        fixed_doc = {}

        for key in doc.keys():
            if key not in fields:
                fixed_doc[key] = doc[key]

        for field in fields:

            for name in doc[field].keys():
                if name in fix.keys():
                    fixed_doc.setdefault(
                        field,
                        {}).setdefault(fix[name], doc[field][name])

        fixed_docs.append(fixed_doc)

    return fixed_docs


def export(docs, out_dir, fix=None, outputs=None, n_jobs=1):
    """ Export data described in documents to fixed folder structure.

        e.g. {out_dir}/{study_name}/subjects/{subject_id}/c_maps/...

        Parameters
        ----------
        out_dir: string
            Destination directory.
        docs: list
            List of documents
        fix: dict
            Map names for c_maps and t_maps
        outputs: dict
            Data to export, default is True for all.
            Possible keys are 'maps', 'data', 'mask', 'model'
            e.g. outputs = {'maps': False} <=> export all but maps
    """
    docs = fix_docs(docs, fix)

    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    if n_jobs == 1:
        for doc in docs:
            _export(doc, out_dir, outputs)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        for doc in docs:
            pool.apply_async(_export, args=(doc, out_dir, outputs))
        pool.close()
        pool.join()


def _export(doc, out_dir, outputs):
    # study_dir = os.path.join(out_dir, doc['study'])
    # subject_dir = os.path.join(study_dir, 'subjects', doc['subject'])

    # if not os.path.exists(study_dir):
    #     os.makedirs(study_dir)
    # if not os.path.exists(subject_dir):
    #     os.makedirs(subject_dir)

    # if outputs is None:
    #     outputs = {}
    # # maps
    # if outputs.get('maps', True):
    #     for dtype in ['t_maps', 'c_maps']:
    #         map_dir = os.path.join(subject_dir, dtype)
    #         if not os.path.exists(map_dir):
    #             os.makedirs(map_dir)
    #         for label, fpath in doc[dtype].iteritems():
    #             img = nb.load(fpath)
    #             fname = '%s.nii.gz' % label.replace(' ', '_')
    #             nb.save(img, os.path.join(map_dir, fname))

    # # fMRI
    # if 'data' in doc and outputs.get('data', True):
    #     for dtype in ['raw_data', 'data']:
    #         img = nb.concat_images(doc[dtype])
    #         fname = 'bold' if dtype == 'data' else 'raw_bold'
    #         nb.save(img, os.path.join(subject_dir, '%s.nii.gz' % fname))

    # mask
    if outputs.get('mask', True):
        img = nb.load(doc['mask'])
        nb.save(img, os.path.join(out_dir, 'mask.nii.gz'))

    # model
    if outputs.get('model', True):
        design_matrix = doc['design_matrix']
        path = os.path.join(out_dir, 'design_matrix.json')
        json.dump(design_matrix, open(path, 'wb'))
        contrasts = doc['contrasts']
        path = os.path.join(out_dir, 'contrasts.json')
        json.dump(contrasts, open(path, 'wb'))


def _get_timeseries(data, row_mask, affine=None):
    if isinstance(data, list):
        return nb.concat_images(np.array(data)[row_mask])
    elif isinstance(data, (str, unicode)):
        img = nb.load(data)
        return nb.Nifti1Image(img.get_data()[row_mask, :], img.get_affine())
    elif isinstance(data, (np.ndarray, np.memmap)):
        if affine is None:
            raise Exception('The affine is not optional '
                            'when data is an array')
        return nb.Nifti1Image(data[row_mask, :], affine)
    else:
        raise ValueError('Data type "%s" not supported' % type(data))


def load_glm_params(doc):
    # for i in xrange(0, len(doc['data']), 100):
    #     print doc['data'][i]

    params = {}

    n_scans = doc['n_scans']
    n_sessions = len(n_scans)

    design_matrix = np.array(doc['design_matrix'])[:, :-n_sessions]

    params['design_matrices'] = []
    params['contrasts'] = []
    params['data'] = []

    offset = 0
    for session_id, scans_count in enumerate(n_scans):
        session_dm = design_matrix[offset:offset + scans_count, :]
        column_mask = ~(np.sum(session_dm, axis=0) == 0)
        row_mask = np.zeros(np.sum(n_scans), dtype=np.bool)
        row_mask[offset:offset + scans_count] = True

        params['design_matrices'].append(session_dm[:, column_mask])

        session_contrasts = {}
        for contrast_id in doc['contrasts']:
            contrast = np.array(doc['contrasts'][contrast_id])
            session_contrast = contrast[column_mask]
            if not np.all(session_contrast == 0):
                session_contrast /= session_contrast.max()
            session_contrasts[contrast_id] = session_contrast
        params['contrasts'].append(session_contrasts)
        params['data'].append(_get_timeseries(doc['data'], row_mask))
        offset += scans_count

    return params


def make_contrasts(params, definitions):
    new_contrasts = []
    for old_session_contrasts in params['contrasts']:
        new_session_contrasts = {}
        for new_contrast_id in definitions:
            contrast = None
            for old_contrast_id in definitions[new_contrast_id]:
                scaler = definitions[new_contrast_id][old_contrast_id]
                con = np.array(old_session_contrasts[old_contrast_id]) * scaler
                if contrast is None:
                    contrast = con
                else:
                    contrast += con
            new_session_contrasts[new_contrast_id] = contrast
        new_contrasts.append(new_session_contrasts)
    return new_contrasts


def execute_glm(doc, out_dir, contrast_definitions=None,
                outputs=None, glm_model='ar1',
                ):
    """Function to execute GLM for one subject --and perhaps multiple
    sessions thereof

    """

    stats_start_time = time.ctime()

    # study_dir = os.path.join(out_dir, doc['study'])

    if outputs is None:
        outputs = {'maps': False,
                   'data': False,
                   'mask': True,
                   'model': True,
                   }
    else:
        outputs['maps'] = False

    subject_id = doc['subject']
    subject_output_dir = os.path.join(
        out_dir, subject_id)

    _export(doc, subject_output_dir, outputs=outputs)

    params = load_glm_params(doc)

    # instantiate GLM
    fmri_glm = FMRILinearModel(params['data'],
                          params['design_matrices'],
                          doc['mask'])

    # fit GLM
    fmri_glm.fit(do_scaling=True, model=glm_model)

    # save beta-maps to disk
    beta_map_dir = os.path.join(subject_output_dir, 'beta_maps')
    if not os.path.exists(beta_map_dir):
        os.makedirs(beta_map_dir)
    for j, glm in zip(xrange(len(fmri_glm.glms)), fmri_glm.glms):
        # XXX save array in some compressed format
        np.savetxt(os.path.join(beta_map_dir, "beta_map_%i.txt" % j),
                   glm.get_beta(),  # array has shape (n_conditions, n_voxels)
                   )

    # define contrasts
    if contrast_definitions is not None:
        params['contrasts'] = make_contrasts(params, contrast_definitions)
    contrasts = sorted(params['contrasts'][0].keys())

    _contrasts = {}
    z_maps = {}

    # compute stats maps
    for index, contrast_id in enumerate(contrasts):
        print ' study[%s] subject[%s] contrast [%s]: %i/%i' % (
            doc['study'], doc['subject'],
            contrast_id, index + 1, len(contrasts)
            )

        contrast = [c[contrast_id] for c in params['contrasts']]
        contrast_name = contrast_id.replace(' ', '_')

        z_map, t_map, c_map, var_map = fmri_glm.contrast(
            contrast,
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True,)

        for dtype, out_map in zip(['z', 't', 'c', 'variance'],
                                  [z_map, t_map, c_map, var_map]):
            map_dir = os.path.join(subject_output_dir, '%s_maps' % dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, '%s.nii.gz' % contrast_name)
            nb.save(out_map, map_path)

            # collect z map
            if dtype == 'z':
                _contrasts[contrast_name] = contrast
                z_maps[contrast_name] = map_path

    # invoke a single API to handle plotting and html business for you
    subject_stats_report_filename = os.path.join(
        subject_output_dir, "report_stats.html")
    glm_reporter.generate_subject_stats_report(
        subject_stats_report_filename,
        _contrasts,
        z_maps,
        doc['mask'],
        design_matrices=list(params['design_matrices']),
        subject_id=doc['subject'],
        cluster_th=15,  # 15 voxels
        start_time=stats_start_time,
        TR=doc['TR'],
        n_scans=doc['n_scans'],
        n_sessions=doc['n_sessions'],
        model=glm_model,
        )

    print "Report for subject %s written to %s" % (
        doc['subject'],
        subject_stats_report_filename)


def inv_perm(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i

    return inverse


def execute_glms(docs, out_dir, contrast_definitions=None,
                 outputs=None, glm_model='ar1', dataset_id=None,
                 do_preproc=True,
                 smoothed=None,
                 n_jobs=None):
    """Function to execute a series of GLMs (one per subject)

    """

    # sanity
    n_jobs = len(docs)
    n_jobs = min(n_jobs, multiprocessing.cpu_count() / 4)

    # preprocess the data
    if do_preproc:
        if not smoothed is None:
            fwhm = smoothed
        else:
            fwhm = smoothed

        subject_fmri_perms = {}

        def subject_factory():
            for doc in docs:
                subject_data = SubjectData()

                subject_data.subject_id = doc['subject']

                # grap anat filename like a ninja
                subject_data.anat = os.path.join(
                    re.search(
                        ".+?%s\/fMRI\/acquisition1" % subject_data.subject_id,
                        doc['raw_data'][0]).group().replace("fMRI", "t1mri"),
                    "anat_%s_3T_neurospin.img" % subject_data.subject_id)

                # don't want no ugly supprises hereafter
                assert os.path.exists(subject_data.anat)

                # grab subject session ids like a ninja
                subject_data.session_id = sorted(list(set(
                            [re.search("\/acquisition1\/(.+?)\/",
                                       x).group(1)
                             for x in doc['raw_data']])))

                # collect list of lists of 3D scans (one list per session)
                perm = []
                subject_data.func = [sorted(
                        [x for x in doc['raw_data'] if s in x])
                                     for s in subject_data.session_id]
                _tmp = [x for session_fmri_files in subject_data.func
                              for x in session_fmri_files]
                for fmri_filename in doc['raw_data']:
                    for k in xrange(len(_tmp)):
                        if fmri_filename == _tmp[k]:
                            perm.append(k)
                            break

                subject_fmri_perms[subject_data.subject_id] = perm

                # set subject output directory (so there'll be no pollution)
                subject_data.output_dir = os.path.join(
                    out_dir, subject_data.subject_id)
                if not os.path.exists(subject_data.output_dir):
                    os.makedirs(subject_data.output_dir)

                # yield input data for this subject
                yield subject_data

        preproc_results = do_subjects_preproc(
        subject_factory(),
        output_dir=out_dir,
        dataset_id=dataset_id,
        fwhm=fwhm,
        n_jobs=n_jobs,
        # do_report=False,
        )

        # sanitize
        assert len(preproc_results) == len(docs)

        for doc in docs:
            for preproc_result in preproc_results:
                if preproc_result['subject_id'] == doc['subject']:

                    # fix shuffled (due to sorting in preproc pipeline)
                    # session-wise fmri files, lest activation maps will be
                    # ultimate garbage
                    doc['data'] = list(np.array([
                            x for session_fmri_files in preproc_result['func']
                            for x in session_fmri_files])[
                        subject_fmri_perms[doc['subject']]])
                    break

    # execute one GLM per subject
    if do_preproc:
        output_dir = out_dir
    else:
        output_dir = os.path.join(out_dir, "not_repreprocessed")
    joblib.Parallel(n_jobs=max(n_jobs / 4, 1))(joblib.delayed(execute_glm)(
            doc,
            output_dir,
            contrast_definitions,
            outputs,
            glm_model,
            ) for doc in docs)
