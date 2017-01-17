import os
import glob
import numpy as np
import pandas as pd
import nibabel
from sklearn.externals.joblib import Parallel, delayed, Memory
from nistats.design_matrix import (make_design_matrix, check_design_matrix)
from pypreprocess.external.nistats.glm import FMRILinearModel
from pypreprocess.reporting.base_reporter import ProgressReport
from pypreprocess.reporting.glm_reporter import (generate_subject_stats_report,
                                                 group_one_sample_t_test)
from nilearn.plotting import plot_prob_atlas, show
import matplotlib.pyplot as plt

n_jobs = int(os.environ.get("N_JOBS", 1))


def _load_contrast_names(cfile):
    contrast_names = []
    with open(cfile, 'r') as fd:
        cons = [x.rstrip("\r\n") for x in fd.readlines()]
        cons = [x for x in cons if x]
        for c in cons:
            c = c.split(" ")
            contrast_names.append(c[1])
    return contrast_names


def _load_condition_keys(cfile):
    conditions = {}
    with open(cfile, 'r') as fd:
        conds = [x.rstrip("\r\n") for x in fd.readlines()]
        conds = [x for x in conds if x]
        for c in conds:
            c = c.split(" ")
            conditions[c[1]] = c[2]
    return conditions


# meta data
tr = 2.
drift_model = 'Cosine'
hrf_model = 'spm + derivative'
hfcut = 128.

data_dir = os.environ.get("DATA_DIR", "/home/elvis/nilearn_data/ds001")
output_dir = os.environ.get("OUTPUT_DIR",
                            os.path.join(data_dir,
                                         "pypreprocess_output/ds001"))
condition_keys = _load_condition_keys(
    os.path.join(data_dir, "models/model001/condition_key.txt"))
contrast_names = _load_contrast_names(
    os.path.join(data_dir, "models/model001/task_contrasts.txt"))
subject_ids = map(os.path.basename,
                  sorted(glob.glob("%s/sub*" % output_dir)))[:8]


def do_subject_glm(subject_id):
    subject_output_dir = os.path.join(output_dir, subject_id)

    # make design matrices
    design_matrices = []
    func = []
    anat = os.path.join(subject_output_dir, "anatomy", "whighres001_brain.nii")
    for run_path in sorted(glob.glob(os.path.join(
            data_dir, subject_id, "model/model001/onsets/task*"))):
        run_id = os.path.basename(run_path)
        run_func = glob.glob(os.path.join(subject_output_dir, "BOLD", run_id,
                                          "wrbold*.nii"))
        assert len(run_func) == 1
        run_func = run_func[0]
        run_onset_paths = sorted(glob.glob(os.path.join(
            data_dir, subject_id, "model/model001/onsets/%s/*" % run_id)))
        onsets = map(np.loadtxt, run_onset_paths)
        conditions = np.hstack(
            [[condition_keys["cond%03i" % (c + 1)]] * len(onsets[c])
             for c in range(len(run_onset_paths))])
        onsets = np.vstack((onsets))
        onsets *= tr
        run_func = nibabel.load(run_func)
        func.append(run_func)
        n_scans = run_func.shape[-1]
        onset, duration, modulation = onsets.T

        frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
        paradigm = pd.DataFrame(dict(name=conditions, onset=onset,
                                     duration=duration, modulation=modulation))
        design_matrix = make_design_matrix(frametimes, paradigm,
                                           hrf_model=hrf_model,
                                           drift_model=drift_model,
                                           period_cut=hfcut)
        design_matrices.append(design_matrix)
    n_runs = len(func)

    # specify contrasts
    _, _, names = check_design_matrix(design_matrix)
    n_columns = len(names)
    contrast_matrix = np.eye(n_columns)
    contrasts = {}
    for c in range(len(condition_keys)):
        contrasts[names[2 * c]] = contrast_matrix[2 * c]
    contrasts["avg"] = np.mean(contrasts.values(), axis=0)

    # more interesting contrasts
    contrasts_ = {}
    for contrast, val in contrasts.items():
        if not contrast == "avg":
            contrasts_["%s_minus_avg" % contrast] = val - contrasts["avg"]
    contrasts = contrasts_

    # fit GLM
    from nilearn.image import smooth_img
    func = smooth_img(func, fwhm=8.)
    print('Fitting a GLM (this takes time)...')
    fmri_glm = FMRILinearModel(func, [check_design_matrix(design_matrix)[1]
                                      for design_matrix in design_matrices],
                               mask='compute')
    fmri_glm.fit(do_scaling=True, model='ar1')

    # save computed mask
    mask_path = os.path.join(subject_output_dir, "mask.nii")
    print("Saving mask image to %s ..." % mask_path)
    nibabel.save(fmri_glm.mask, mask_path)

    # compute contrast maps
    z_maps = {}
    effects_maps = {}
    for contrast_id, contrast_val in contrasts.items():
        print("\tcontrast id: %s" % contrast_id)
        z_map, t_map, effects_map, var_map = fmri_glm.contrast(
            [contrast_val] * n_runs, con_id=contrast_id, output_z=True,
            output_stat=True, output_effects=True, output_variance=True)
        for map_type, out_map in zip(['z', 't', 'effects', 'variance'],
                                     [z_map, t_map, effects_map, var_map]):
            map_dir = os.path.join(subject_output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            print("\t\tWriting %s ..." % map_path)
            nibabel.save(out_map, map_path)
            if map_type == 'z':
                z_maps[contrast_id] = map_path
            if map_type == 'effects':
                effects_maps[contrast_id] = map_path

    # # generate stats report
    # stats_report_filename = os.path.join(subject_output_dir, "reports",
    #                                      "report_stats.html")
    # generate_subject_stats_report(
    #     stats_report_filename, contrasts, z_maps, fmri_glm.mask, anat=anat,
    #     threshold=2.3, cluster_th=15, design_matrices=design_matrices, TR=tr,
    #     subject_id="sub001", n_scans=n_scans, hfcut=hfcut,
    #     paradigm=paradigm, frametimes=frametimes,
    #     drift_model=drift_model, hrf_model=hrf_model)
    # ProgressReport().finish_dir(subject_output_dir)

    return dict(subject_id=subject_id, mask=mask_path,
                effects_maps=effects_maps, z_maps=z_maps, contrasts=contrasts)


# first level GLM
mem = Memory(os.path.join(output_dir, "cache_dir"))
n_jobs = min(n_jobs, len(subject_ids))
first_levels = Parallel(n_jobs=n_jobs)(delayed(mem.cache(do_subject_glm))(
    subject_id) for subject_id in subject_ids)

# run second-level GLM
group_zmaps = group_one_sample_t_test(
    [subject_data["mask"] for subject_data in first_levels],
    [subject_data["effects_maps"] for subject_data in first_levels],
    first_levels[0]["contrasts"],
    output_dir, threshold=2.)
plot_prob_atlas([zmap for zmap in group_zmaps.values() if "_minus_" in zmap],
                threshold=1.2, view_type="filled_contours")
plt.savefig("group_zmaps.png")
show()
