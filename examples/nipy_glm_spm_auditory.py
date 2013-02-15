"""
:Synopsis: Demo script for nipy's GLM tools
:Author: dohmatob elvis dopgima

"""

import sys
import os
import pylab as pl
import numpy as np
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs import viz
import nibabel

"""chdir to script dir"""
os.chdir(os.path.dirname(sys.argv[0]))

sys.path.append("..")
import nipype_preproc_spm_utils
from datasets_extras import fetch_spm_auditory_data
from io_utils import do_3Dto4D_merge

DATASET_DESCRIPTION = """\
<p>MoAEpilot <a href="http://www.fil.ion.ucl.ac.uk/spm/data/auditory/">\
SPM auditory dataset</a>.</p>\
"""

if len(sys.argv)  < 3:
    print ("\r\nUsage: python %s <spm_auditory_MoAEpilot_dir>"
           " <output_dir>\r\n") % sys.argv[0]
    sys.exit(1)

"""set data dir"""
DATA_DIR = os.path.abspath(sys.argv[1])

# set output dir
OUTPUT_DIR = os.path.abspath(sys.argv[2])
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


"""fetch spm auditory data"""
_subject_data = fetch_spm_auditory_data(DATA_DIR)

subject_data = nipype_preproc_spm_utils.SubjectData()
subject_data.func = _subject_data["func"]
subject_data.anat = _subject_data["anat"]
subject_data.output_dir = os.path.join(
    OUTPUT_DIR, subject_data.subject_id)

"""preprocess the data"""
report_filename = os.path.join(OUTPUT_DIR,
                               "_report.html")
results = nipype_preproc_spm_utils.do_subjects_preproc(
    [subject_data],
    do_report=False,
    do_deleteorient=False,
    fwhm=[6, 6, 6],
    dataset_description=DATASET_DESCRIPTION,
    report_filename=report_filename)

"""collect preprocessed data"""
fmri_data = results[0]['func']
fmri_data = do_3Dto4D_merge(fmri_data)

"""construct experimental paradigm"""
tr = 7
n_scans = 96
_duration = 6
epoch_duration = _duration * tr
conditions = ['rest', 'active'] * 8
duration = epoch_duration * np.ones(len(conditions))
onset = np.linspace(0, (len(conditions) - 1) * epoch_duration,
                    len(conditions))
paradigm = BlockParadigm(con_id=conditions, onset=onset, duration=duration)
hfcut = 2 * 2 * epoch_duration

"""construct design matrix"""
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
design_matrix = make_dmtx(frametimes,
                          paradigm, hrf_model=hrf_model,
                          drift_model=drift_model, hfcut=hfcut)

"""show design matrix"""
ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')
pl.savefig(os.path.join(OUTPUT_DIR, 'design_matrix.png'))

"""specify contrasts"""
contrasts = {}
n_columns = len(design_matrix.names)
for i in xrange(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

# more interesting contrasts
contrasts['active-rest'] = contrasts['active'] - contrasts['rest']

# fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FMRILinearModel(fmri_data, design_matrix.matrix,
           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

"""save computed mask"""
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)

"""level 1 (within subject) inference"""
print "Computing contrasts .."
for contrast_id in contrasts:
    print "\tcontrast id: %s" % contrast_id
    z_map, t_map, eff_map, var_map = fmri_glm.contrast(
        contrasts[contrast_id],
        con_id=contrast_id,
        output_z=True,
        output_stat=True,
        output_effects=True,
        output_variance=True,)

    if contrast_id == 'active-rest':
        z_threshold = 5
        pos_data = z_map.get_data() * (z_map.get_data() > 0)
        vmax = pos_data.max()
        vmin = - vmax

        viz.plot_map(pos_data, z_map.get_affine(),
                     cmap=pl.cm.spectral,
                     vmin=vmin,
                     vmax=vmax,
                     threshold=z_threshold)

    for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                              [z_map, t_map, eff_map, var_map]):
        map_dir = os.path.join(
            subject_data.output_dir, '%s_maps' % dtype)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(
            map_dir, '%s.nii.gz' % contrast_id)
        nibabel.save(out_map, map_path)

        print "\t\t%s map: %s" % (dtype, map_path)

    print

"""show all generated plots this far"""
pl.show()
