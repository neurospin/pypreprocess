"""
GLM utils (backended by nipy).

WIP
===
Experimenting the possibility of specifying timing info in .ini files
alongside info specific to preprocessing business
XXX TODO: take into account multi-models (a model is simply a
one-to-one mapping a subset of bold files, unto onset / timing
files.


"""
# Author: Elvis DOHMATOB, Christophe PALIER

import os
import glob
import numpy as np
import scipy.io
import nibabel
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.glm import FMRILinearModel
from pypreprocess.datasets import fetch_spm_multimodal_fmri_data
from pypreprocess.purepython_preproc_utils import do_subject_preproc
from pypreprocess.external.joblib import Memory
from pypreprocess.reslice import reslice_vols


def parse_onset_file(onset_file):
    """Parses onset file.

    Lines of the file must be of the form:

        condition_name onset duration amplitude

    The duration and amplitude columns are optional.

    Returns
    -------
    conditions: list of strings
        The names of the presentations corresponding to the onset times.

    onsets: array of length len(conditons)
        The onsets.

    durations: array of length len(conditions)
        The duration of each stimulus presentation (one per onset time)

    amplitudes: array of length len(conditions)
        The amplitude of each stimulus presentation (one per onset time)
    """
    conditions = []
    onsets = []
    durations = []
    amplitudes = []
    line_cnt = 0
    fd = open(onset_file, 'r')
    while True:
        line = fd.readline()
        if not line:
            break
        line = line.rstrip("\r\n")
        line = line.split(" ")
        line_cnt += 1
        if len(line) not in [2, 3, 4]:
            raise ValueError("Mal-formed line %i: %s" % (
                    line_cnt, " ".join(line)))
        if len(line) == 2:
            line.append(0.)
        if len(line) == 3:
            line.append(1.)
        condition, onset, duration, amplitude = line
        conditions.append(condition)
        onsets.append(float(onset))
        durations.append(float(duration))
        amplitudes.append(float(amplitude))

    fd.close()
    if not line_cnt > 0:
        raise ValueError(
            "Couldn't read any data from onset file: %s" % onset_file)
    return map(np.array, [conditions, onsets, durations, amplitudes])

# fetch data
data_dir = "examples/spm_multimodal/"
subject_data = fetch_spm_multimodal_fmri_data(data_dir)

# XXX to be verified
tr = 2.
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
hfcut = 128.
time_units = "tr"  # default if 1
if time_units == "tr":
    time_units = tr

# re-write onset files into compatible format
for sess in xrange(2):
    trials = getattr(subject_data, "trials_ses%i" % (sess + 1))
    fd = open(trials.split(".")[0] + ".txt", 'w')
    timing = scipy.io.loadmat(trials, squeeze_me=True, struct_as_record=False)
    onsets = np.hstack(timing['onsets'])
    durations = np.hstack(timing['durations'])
    amplitudes = np.ones_like(onsets)
    conditions = [list(timing['names'][i:i + 1]) * len(timing['onsets'][i])
                  for i in xrange(len(timing['names']))]
    conditions = np.hstack(conditions)
    assert len(amplitudes) == len(onsets) == len(durations) == len(conditions)
    for condition, onset, duration, amplitude in zip(conditions, onsets,
                                                     durations, amplitudes):
        fd.write("%s %s %s %s\r\n" % (condition, onset, duration, amplitude))
    fd.close()

output_dir = ""
anat_wildcard = 'sMRI/smri.img'
session_1_onset = "fMRI/trials_ses1.txt"
session_1_func = "fMRI/Session1/fMETHODS-0005-00*.img"
session_2_onset = "fMRI/trials_ses2.txt"
session_2_func = "fMRI/Session2/fMETHODS-0006-00*.img"

subject_dirs = sorted(glob.glob("%s/sub*" % data_dir))
session_onset_wildcards = [session_1_onset, session_2_onset]
session_func_wildcards = [session_1_func, session_2_func]


def do_subject_glm(subject_data):
    """FE analysis for a single subject."""
    subject_id = subject_data['subject_id']
    output_dir = subject_data["output_dir"]
    func_files = subject_data['func']
    anat = subject_data['anat']
    onset_files = subject_data['onset']
    mem = Memory(os.path.join(output_dir, "cache"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if 0:
        subject_data = mem.cache(do_subject_preproc)(
            dict(func=func_files, anat=anat, output_dir=output_dir))
        func_files = subject_data['func']
        anat = subject_data['anat']

        # reslice func images
        func_files = [mem.cache(reslice_vols)(
                sess_func,
                target_affine=nibabel.load(sess_func[0]).get_affine())
                  for sess_func in func_files]

    ### GLM: loop on (session_bold, onse_file) pairs over the various sessions
    design_matrices = []
    for session, (func_file, onset_file) in enumerate(zip(func_files,
                                                          onset_files)):
        if isinstance(func_file, str):
            bold = nibabel.load(func_file)
        else:
            if len(func_file) == 1:
                func_file = func_file[0]
                bold = nibabel.load(func_file)
                assert len(bold.shape) == 4
                n_scans = bold.shape[-1]
                del bold
            else:
                n_scans = len(func_file)
        frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
        conditions, onsets, durations, amplitudes = parse_onset_file(
            onset_file)
        onsets *= tr
        durations *= tr
        paradigm = BlockParadigm(con_id=conditions, onset=onsets,
                                 duration=durations, amplitude=amplitudes)
        design_matrices.append(make_dmtx(
                frametimes,
                paradigm, hrf_model=hrf_model,
                drift_model=drift_model, hfcut=hfcut))

    # specify contrasts
    n_columns = len(design_matrices[0].names)
    contrasts = {}
    for i in xrange(paradigm.n_conditions):
        contrasts['%s' % design_matrices[0].names[2 * i]
                  ] = np.eye(n_columns)[2 * i]

    # more interesting contrasts
    contrasts['faces-scrambled'] = contrasts['faces'
                                             ] - contrasts['scrambled']
    contrasts['scrambled-faces'] = -contrasts['faces-scrambled']
    contrasts['effects_of_interest'] = contrasts['faces'
                                                 ] + contrasts['scrambled']

    # effects of interest F-test
    diff_contrasts = []
    for i in xrange(paradigm.n_conditions - 1):
        a = contrasts[design_matrices[0].names[2 * i]]
        b = contrasts[design_matrices[0].names[2 * (i + 1)]]
        diff_contrasts.append(a - b)
    contrasts["diff"] = diff_contrasts

    # fit GLM
    print 'Fitting a GLM (this takes time)...'
    fmri_glm = FMRILinearModel([nibabel.concat_images(sess_func,
                                                      check_affines=False)
                                for sess_func in func_files],
                               [design_matrix.matrix
                                for design_matrix in design_matrices],
                               mask='compute'
                               )
    fmri_glm.fit(do_scaling=True, model='ar1')

    # save computed mask
    mask_path = os.path.join(output_dir, "mask.nii.gz")

    print "Saving mask image %s" % mask_path
    nibabel.save(fmri_glm.mask, mask_path)

    # compute contrasts
    z_maps = {}
    effects_maps = {}
    for contrast_id, contrast_val in contrasts.iteritems():
        print "\tcontrast id: %s" % contrast_id
        if np.ndim(contrast_val) > 1:
            contrast_type = "t"
        else:
            contrast_type = "F"
        z_map, t_map, effects_map, var_map = fmri_glm.contrast(
            [contrast_val] * 2,
            con_id=contrast_id,
            contrast_type=contrast_type,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True
            )

        # store stat maps to disk
        for map_type, out_map in zip(['z', 't', 'effects', 'variance'],
                                  [z_map, t_map, effects_map, var_map]):
            map_dir = os.path.join(
                output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            print "\t\tWriting %s ..." % map_path
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if map_type == 'z':
                z_maps[contrast_id] = map_path
            if map_type == 'effects':
                effects_maps[contrast_id] = map_path

    return subject_id, anat, effects_maps, z_maps, contrasts, fmri_glm.mask


if __name__ == "__maih__":
    mem = Memory(os.path.join(output_dir, "cache"))
    first_level_glms = map(mem.cache(do_subject_glm), subject_dirs)

    # plot stats (per subject)
    import matplotlib.pyplot as plt
    import nipy.labs.viz as viz
    all_masks = []
    all_effects_maps = []
    for (subject_id, anat, effects_maps, z_maps,
         contrasts, mask) in first_level_glms:
        all_masks.append(mask)
        anat_img = nibabel.load(anat)
        z_map = nibabel.load(z_maps.values()[0])
        all_effects_maps.append(effects_maps)
        for contrast_id, z_map in z_maps.iteritems():
            z_map = nibabel.load(z_map)
            viz.plot_map(z_map.get_data(), z_map.get_affine(),
                         anat=anat_img.get_data(),
                         anat_affine=anat_img.get_affine(), slicer='ortho',
                         title="%s: %s" % (subject_id, contrast_id),
                         black_bg=True, cmap=viz.cm.cold_hot, threshold=2.3)
            plt.savefig("%s_%s.png" % (subject_id, contrast_id))
