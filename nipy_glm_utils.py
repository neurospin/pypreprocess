import os
import warnings
import multiprocessing

import nibabel as nb
import numpy as np

from nipy.modalities.fmri.glm import FMRILinearModel
from external.nisl import resampling


def apply_glm(dataset_id, out_dir, data, params, resample=True, n_jobs=1):
    out_dir = os.path.join(out_dir, dataset_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    out_dir = os.path.join(out_dir, 'subjects')

    pool = multiprocessing.Pool(n_jobs)

    stat_maps = {}
    for doc in params:
        subject_id = doc['subject_id']
        if subject_id not in data:
            warnings.warn('Missing data for %s, '
                          'subject is skipped.' % subject_id)
            continue
        if n_jobs == 1:
            stat_maps[subject_id] = _apply_glm(
                os.path.join(out_dir, subject_id), data[subject_id],
                doc['design_matrices'], doc['contrasts'],
                resample=resample)
        else:
            stat_maps[subject_id] = pool.apply_async(
                _apply_glm,
                args=(os.path.join(out_dir, subject_id), data[subject_id],
                      doc['design_matrices'], doc['contrasts']),
                kwds={'resample': resample})

    pool.close()
    pool.join()

    return stat_maps


def _apply_glm(out_dir, data, design_matrices,
               contrasts, mask='compute', model_id=None, resample=True):

    # print out_dir
    bold_dir = os.path.join(out_dir, 'fmri')
    if not os.path.exists(bold_dir):
        os.makedirs(bold_dir)
    for i, img in enumerate(data):
        if type(img) is str:
            img = nb.load(img)
        nb.save(img, os.path.join(bold_dir, 'bold_session_%i.nii.gz' % i))

    # fit glm
    glm = FMRILinearModel(data, design_matrices, mask=mask)
    glm.fit(do_scaling=True, model='ar1')

    nb.save(glm.mask, os.path.join(out_dir, 'mask.nii.gz'))
    if resample:
        resample_niimg(os.path.join(out_dir, 'mask.nii.gz'))

    stat_maps = {}
    for contrast_id in contrasts:
        stat_maps[contrast_id] = {}
        z_map, t_map, c_map, var_map = glm.contrast(
            contrasts[contrast_id],
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True,)

        for dtype, out_map in zip(['z', 't', 'c', 'variance'],
                                  [z_map, t_map, c_map, var_map]):
            map_dir = os.path.join(out_dir, '%s_maps' % dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            if model_id:
                map_path = os.path.join(
                    map_dir, '%s_%s.nii.gz' % (model_id, contrast_id))
            else:
                map_path = os.path.join(
                    map_dir, '%s.nii.gz' % contrast_id)
            nb.save(out_map, map_path)
            if resample:
                resample_niimg(map_path)

            stat_maps[contrast_id][dtype] = map_path


    return stat_maps


def resample_niimg(niimg):
    target_affine = np.array([[-3., 0., 0., 78.],
                              [0., 3., 0., -111.],
                              [0., 0., 3., -51.],
                              [0., 0., 0., 1., ]])
    target_shape = (53, 63, 46)

    if (_get_shape(niimg) != target_shape or
        _get_affine(niimg) != target_affine):
        img = resampling.resample_img(niimg, target_affine, target_shape)
        nb.save(img, niimg)


def _get_affine(niimg):
    if hasattr(niimg, 'get_affine'):
        return niimg.get_affine()
    elif os.path.exists(niimg):
        return nb.load(niimg).get_affine()
    else:
        raise Exception('`niimg` must either be a valid '
                        'image path or an object exposing the '
                        '`get_affine` method. Got %s instead' % niimg)


def _get_shape(niimg):
    if hasattr(niimg, 'shape'):
        return niimg.shape
    elif os.path.exists(niimg):
        return nb.load(niimg).shape
    else:
        raise Exception('`niimg` must either be a valid '
                        'image path or an object having a '
                        '`shape` attribute. Got %s instead' % niimg)
