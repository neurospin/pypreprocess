import os
import sys
import shutil

# local imports
from utils import apply_preproc, load_preproc, load_glm_params

# parent dir imports
sys.path.append('..')

from nipy_glm_utils import apply_glm
from datasets_extras import fetch_openfmri

FULL_ID = 'ds000101'
SHORT_ID = 'ds101'
NAME = 'Simon task dataset'
DESCRIPTION = """
The "NYU Simon Task" dataset comprises data collected from 21 healthy adults
while they performed a rapid event-related Simon task.
**Please note that all data have been uploaded regardless of quality-
it is up to the user to check for data quality (movement etc).

On each trial (inter-trial interval (ITI) was 2.5 seconds, with null
events for jitter), a red or green box appeared on the right or left
side of the screen. Participants used their left index finger to respond
to the presentation of a green box, and their right index finger to
respond to the presentation of a red box.In congruent trials the green
box appeared on the left or the red box on the right, while in more
demanding incongruent trials the green box appeared on the right and
the red on the left.

Subjects performed two blocks, each containing 48 congruent and 48
incongruent trials, presented in a pre-determined order (as per OptSeq),
interspersed with 24 null trials (fixation only).

Functional imaging data were acquired using a research dedicated Siemens
Allegra 3.0 T scanner, with a standard Siemens head coil, located at the
NYU Center for Brain Imaging.

We obtained 151 contiguous echo planar imaging (EPI) whole-brain
functional volumes (TR=2000 ms; TE=30 ms; flip angle=80, 40 slices,
matrix=64x64; FOV=192 mm; acquisition voxel size=3x3x4mm) during each
of the two simon task blocks. A high-resolution T1-weighted anatomical
image was also acquired using a magnetization prepared gradient echo
sequence (MPRAGE, TR=2500 ms; TE=3.93 ms; TI=900 ms; flip angle=8;
176 slices, FOV=256 mm).

These data have not been published previously.

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
