import os
import sys
import shutil

# local imports
from utils import apply_preproc, load_preproc, load_glm_params

# parent dir imports
sys.path.append('..')

from nipy_glm_utils import apply_glm
from datasets_extras import fetch_openfmri

FULL_ID = 'ds000105'
SHORT_ID = 'ds105'
NAME = 'Visual object recognition'
DESCRIPTION = """
Neural responses, as reflected in hemodynamic changes, were measured in
six subjects (five female and one male) with gradient echo echoplanar
imaging on a GE 3T scanner (General Electric, Milwaukee, WI) [repetition
time (TR) = 2500 ms, 40 3.5-mm-thick sagittal images, field of view
(FOV) = 24 cm, echo time (TE) = 30 ms, flip angle = 90] while they
performed a one-back repetition detection task. High-resolution
T1-weighted spoiled gradient recall (SPGR) images were obtained for
each subject to provide detailed anatomy (124 1.2-mm-thick sagittal
images, FOV = 24 cm). Stimuli were gray-scale images of faces,
houses, cats, bottles, scissors, shoes, chairs, and nonsense
patterns. The categories were chosen so that all stimuli from a
given category would have the same base level name. The specific
categories were selected to allow comparison with our previous
studies (faces, houses, chairs, animals, and tools) or ongoing
studies (shoes and bottles). Control nonsense patterns were
phase-scrambled images of the intact objects. Twelve time series
were obtained in each subject. Each time series began and ended
with 12 s of rest and contained eight stimulus blocks of 24-s
duration, one for each category, separated by 12-s intervals of
rest. Stimuli were presented for 500 ms with an interstimulus
interval of 1500 ms. Repetitions of meaningful stimuli were pictures
of the same face or object photographed from different angles. Stimuli
for each meaningful category were four images each of 12 different exemplars.

Get full description <a href="https://openfmri.org/dataset/ds000105">\
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
