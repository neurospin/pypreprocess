import os
import sys
import shutil

# local imports
from utils import apply_preproc, load_preproc, load_glm_params

# parent dir imports
sys.path.append('..')

from nipy_glm_utils import apply_glm
from datasets_extras import fetch_openfmri

FULL_ID = 'ds000102'
SHORT_ID = 'ds102'
NAME = 'Flanker task (event-related)'
DESCRIPTION = """
The "NYU Slow Flanker" dataset comprises data collected from 26 healthy
adults while they performed a slow event-related Eriksen Flanker task.
**Please note that all data have been uploaded regardless of quality- it
is up to the user to check for data quality (movement etc).

On each trial (inter-trial interval (ITI) varied between 8 s and 14 s;
mean ITI=12 s),participants used one of two buttons on a response pad
to indicate the direction of a central arrow in an array of 5 arrows.
In congruent trials the flanking arrows pointed in the same direction
as the central arrow (e.g., < < < < <), while in more demanding
incongruent trials the flanking arrows pointed in the opposite
direction (e.g., < < > < <).

Subjects performed two 5-minute blocks, each containing 12 congruent
and 12 incongruent trials, presented in a pseudorandom order.

Functional imaging data were acquired using a research dedicated Siemens
Allegra 3.0 T scanner, with a standard Siemens head coil, located at
theNYU Center for Brain Imaging.

We obtained 146 contiguous echo planar imaging (EPI) whole-brain functional
volumes (TR=2000 ms; TE=30 ms; flip angle=80, 40 slices, matrix=64x64;
FOV=192 mm; acquisition voxel size=3x3x4mm) during each of the two flanker
task blocks. A high-resolution T1-weighted anatomical image was also
acquired using a magnetization prepared gradient echo sequence
(MPRAGE, TR=2500 ms; TE=3.93 ms; TI=900 ms; flip angle=8;
176 slices, FOV=256 mm).

Get full description <a href="https://openfmri.org/dataset/ds000101">\
here</a>.\
"""

MODEL_ID = 'model001'

ignore_list = []


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print ("Usage: python %s <data_root_dir> "
        "<preproc_root_dir> <glm_root_dir>" % sys.argv[0])
        print ("Example:\r\npython %s ~/datasets/raw"
               " ~/datasets/preproc ~/datasets/glm") % sys.argv[0]
        sys.exit(1)

    root_dir, preproc_dir, glm_dir = sys.argv[1:]

    # download data
    data_dir = fetch_openfmri(FULL_ID, root_dir)

    # alternative task_contrasts
    contrasts_file = '%s_task_contrasts.txt' % SHORT_ID
    assert os.path.isfile(contrasts_file), \
        "No contrasts file: %s" % contrasts_file
    dest = os.path.join(data_dir, SHORT_ID,
                        'models', MODEL_ID, 'task_contrasts.txt')

    shutil.copy(contrasts_file, dest)

    # apply SPM preprocessing
    apply_preproc(SHORT_ID, data_dir, preproc_dir, ignore_list,
                  dataset_description=DESCRIPTION)

    # prepare GLM (get data and design)
    preproc_data, motion_params = load_preproc(SHORT_ID, preproc_dir)

    glm_params = load_glm_params(SHORT_ID, data_dir, MODEL_ID,
                                 subject_ids=preproc_data.keys(),
                                 motion_params=motion_params)

    apply_glm(SHORT_ID, glm_dir, preproc_data,
              glm_params, resample=True, n_jobs=-1)
