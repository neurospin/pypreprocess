import os
import time
import numpy as np
import nibabel
from sklearn.externals.joblib import Memory, Parallel, delayed
from pypreprocess.external.nistats.glm import FirstLevelGLM
from pypreprocess.reporting.base_reporter import ProgressReport
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.fsl_to_nistats import (read_fsl_design_file,
                                         make_dmtx_from_timing_files)
# from pypreprocess.reporting.glm_reporter import group_one_sample_t_test

# sanitize directories
root = os.environ.get("ROOT", "/")
experiment_dir = os.path.join(root, "storage/data/HCP/S500-1")
data_dir = experiment_dir
output_dir = "GLM"  # XXX change to have more control
for place in [experiment_dir, data_dir]:
    if not os.path.isdir(place):
        raise RuntimeError("%s doesn't exist! Export ROOT variable" % (
            place))

tr = .72
# hrf_model = "Canonical with Derivative"
hrf_model = "spm + derivative"
drift_model = "Cosine"
hfcut = 100.
mem = Memory(os.path.join(output_dir, "cache_dir"))
# cons = [
#     "0BK-2BK", "PLACE-AVG", "FACE-AVG", "TOOL-AVG", "BODY-AVG",
#     "LH-RH", "LF-RF", "T-AVG",
#     "FACES-SHAPES",
#     "TOM-RANDOM",
#     "MATH-STORY"]
cons = ["LH-RH", "RH-LH", "RF-LF", "LF-RF", "T-AVG"]


def do_subject_glm(subject_id, task, cons, memory=None, smoothing_fwhm=0.,
                   directions=None, report=True):
    stats_start_time = time.ctime()
    if directions is None:
        directions = ['LR', 'RL']
    subject_dir = os.path.join(experiment_dir, subject_id)
    subject_output_dir = os.path.join(output_dir, subject_id)
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)
    fmri_files = [os.path.join(data_dir, subject_id,
                               "MNINonLinear/Results/",
                               "tfMRI_%s_%s/tfMRI_%s_%s.nii.gz" % (
                                   task, direction, task, direction))
                  for direction in directions]
    anat_file = os.path.join(data_dir, subject_id,
                             "MNINonLinear/T1w_restore_brain.nii.gz")
    design_files = [os.path.join(subject_dir,
                                 "MNINonLinear/Results/tfMRI_%s_%s/",
                                 "tfMRI_%s_%s_hp200_s4_level1.fsf"
                                 ) % (task, direction, task, direction)
                    for direction in directions]

    for x in fmri_files:
        if not os.path.exists(x):
            print("%s is missing; skipping subject %s ..." % (x, subject_id))
            return
    assert len(fmri_files) == len(design_files)

    # the actual GLM stuff
    n_scans = []
    frametimes_list = []
    paradigms = []
    design_matrices = []
    for _, (direction, fmri_file, design_file) in enumerate(
            zip(directions, fmri_files, design_files)):
        if not os.path.isfile(design_file):
            print "Can't find design file %s; skipping subject %s" % (
                design_file, subject_id)
            return

        # read the experimental setup
        print "Reading experimental setup from %s ..." % design_file
        fsl_condition_ids, timing_files, fsl_contrast_ids, contrast_values = \
            read_fsl_design_file(design_file)

        # fix timing filenames
        timing_files = [tf.replace("EVs", "tfMRI_%s_%s/EVs" % (
            task, direction)) for tf in timing_files]

        # make design matrix
        print "Constructing design matrix for direction %s ..." % direction
        _n_scans = nibabel.load(fmri_file).shape[-1]
        n_scans.append(_n_scans)
        design_matrix, paradigm, frametimes = make_dmtx_from_timing_files(
            timing_files, fsl_condition_ids, n_scans=_n_scans, tr=tr,
            hrf_model=hrf_model, drift_model=drift_model, period_cut=hfcut)
        paradigms.append(paradigm)
        frametimes_list.append(frametimes)
        design_matrices.append(design_matrix)

        # convert contrasts to dict
        contrasts = dict((contrast_id,
                          # append zeros to end of contrast to match design
                          np.hstack((contrast_value, np.zeros(len(
                              design_matrix.columns) - len(contrast_value)))))

                         for contrast_id, contrast_value in zip(
                             fsl_contrast_ids, contrast_values))

        # more interesting contrasts
        if task == 'MOTOR':
            contrasts['RH-LH'] = contrasts['RH'] - contrasts['LH']
            contrasts['LH-RH'] = -contrasts['RH-LH']
            contrasts['RF-LF'] = contrasts['RF'] - contrasts['LF']
            contrasts['LF-RF'] = -contrasts['RF-LF']
            contrasts['H'] = contrasts['RH'] + contrasts['LH']
            contrasts['F'] = contrasts['RF'] + contrasts['LF']
            contrasts['H-F'] = contrasts['RH'] + contrasts['LH'] - (
                contrasts['RF'] - contrasts['LF'])
            contrasts['F-H'] = -contrasts['H-F']

        contrasts = dict((k, v) for k, v in contrasts.iteritems() if "-" in k)
        contrasts = dict((k, v) for k, v in contrasts.iteritems() if k in cons)

    # replicate contrasts across sessions
    contrasts = dict((cid, [cval] * len(fmri_files))
                     for cid, cval in contrasts.iteritems())

    print (('Fitting a "Fixed Effect" GLM for merging LR and RL '
            'phase-encoding directions for subject %s (%s task)...' % (
                subject_id, task)))
    fmri_glm = FirstLevelGLM(memory=memory, smoothing_fwhm=smoothing_fwhm)
    fmri_glm.fit(fmri_files, design_matrices)

    # save computed mask
    mask_path = os.path.join(subject_output_dir, "mask.nii")
    print("Saving mask image to %s ..." % mask_path)
    fmri_glm.masker_.mask_img_.to_filename(mask_path)

    # do the actual model fit
    z_maps = {}
    effects_maps = {}
    map_dirs = {}
    for contrast_id, contrast_val in contrasts.iteritems():
        print("\tcontrast id: %s" % contrast_id)
        z_map, eff_map = fmri_glm.transform(
            contrast_val, contrast_name=contrast_id, output_z=True,
            output_effects=True)

        # store stat maps to disk
        for map_type, out_map in zip(['z', 'effects'],
                                     [z_map, eff_map]):
            map_dir = os.path.join(
                subject_output_dir, '%s_maps' % map_type)
            map_dirs[map_type] = map_dir
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, '%s_%s.nii' % (map_type,
                                                            contrast_id))
            print("\t\tWriting %s ..." % map_path)
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if map_type == 'z':
                z_maps[contrast_id] = map_path

            if map_type == 'effects':
                effects_maps[contrast_id] = map_path

    if report:
        stats_report_filename = os.path.join(subject_output_dir, "reports",
                                             "report_stats.html")
        generate_subject_stats_report(
            stats_report_filename, contrasts, z_maps,
            fmri_glm.masker_.mask_img_, threshold=3., cluster_th=15,
            design_matrices=design_matrices, TR=tr, subject_id=subject_id,
            start_time=stats_start_time, n_scans=n_scans, paradigm=paradigm,
            frametimes=frametimes, drift_model=drift_model, hfcut=hfcut,
            title="GLM for subject %s" % subject_id, hrf_model=hrf_model,
            anat=anat_file)
        ProgressReport().finish_dir(subject_output_dir)
        print("Statistic report written to %s\r\n" % stats_report_filename)

    print ("Done (subject %s)" % subject_id)
    return dict(subject_id=subject_id, mask=mask_path,
                effects_maps=effects_maps, z_maps=z_maps,
                contrasts=contrasts)


if __name__ == "__main__":
    # get subject ids to process
    subject_ids = ['994273', '140117', '139637', '139233', '138534', '138231',
                   '137936', '137633', '137128', '137027', '136833', '136227',
                   '135932', '135528', '135225', '134324', '133928', '133827',
                   '133625', '133019', '132118', '131924', '131722', '131217',
                   '130922', '130316', '130013', '129028', '128632', '128127',
                   '127933', '127630', '126628', '126325', '125525', '124826',
                   '124422', '124220', '123925', '123420', '123117', '122620',
                   '122317', '121618', '121315', '120515', '120212', '120111',
                   '119833', '118932', '118730', '118528', '117324', '117122',
                   '116524', '114924', '114419', '113922', '113821', '113619',
                   '113215', '112819', '111716', '111413', '111312', '110411',
                   '109325', '109123', '108828', '108525', '108323', '108121',
                   '107422', '107321', '106521', '106319', '106016', '105216',
                   '105014', '104820', '103818', '103515', '103414', '103111',
                   '102816', '102311', '102008', '101915', '101410', '101309',
                   '101107']
    for bad in ['105115', '100408', '124422', '113619', '133625', '114924']:
        if bad in subject_ids:
            subject_ids.remove(bad)
    subject_ids = ["100307"]

    for task in ["MOTOR"]:
        # run first-level GLM
        n_jobs = min(os.environ.get("N_JOBS", len(subject_ids)),
                     len(subject_ids))
        first_levels = Parallel(n_jobs=n_jobs)(delayed(do_subject_glm)(
            subject_id, task, cons, memory=mem)
            for subject_id in subject_ids if subject_id not in [])
        first_levels = [x for x in first_levels if x is not None]
        print(task, len(first_levels))

        # # run second-level GLM
        # output_dir = os.path.join(data_dir, "GLM%s" % (
        #     ["DC+SBRef", "DC+LoG"][pipeline]))
        # mem.cache(group_one_sample_t_test)(
        #     [subject_data["mask"] for subject_data in first_levels],
        #     [subject_data["effects_maps"] for subject_data in first_levels],
        #     first_levels[0]["contrasts"],
        #     output_dir)