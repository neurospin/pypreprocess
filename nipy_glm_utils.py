import os
import multiprocessing

import nibabel as nb

from nipy.modalities.fmri.glm import FMRILinearModel


def apply_glm(out_dir, data, params, n_jobs=1):
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    pool = multiprocessing.Pool(n_jobs)

    for doc in params:
        subject_id = doc['subject']

        if n_jobs == 1:
            _apply_glm(os.path.join(out_dir, subject_id), data[subject_id],
                       doc['design_matrices'], doc['contrasts'])
        else:
            pool.apply_async(
                _apply_glm,
                args=(os.path.join(out_dir, subject_id), data[subject_id],
                      doc['design_matrices'], doc['contrasts']))

    pool.close()
    pool.join()


def _apply_glm(out_dir, data, design_matrices,
               contrasts, mask='compute', model_id=None):

    print out_dir

    bold_dir = os.path.join(out_dir, 'frmi')
    if not os.path.exists(bold_dir):
        os.makedirs(bold_dir)
    for i, img in enumerate(data):
        nb.save(img, os.path.join(bold_dir, 'bold_session_%i.nii.gz' % i))

    # fit glm
    glm = FMRILinearModel(data, design_matrices, mask=mask)
    glm.fit(do_scaling=True, model='ar1')

    for contrast_id in contrasts:
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
