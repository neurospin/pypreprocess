import os
import sys
import shutil

# local imports
from utils import apply_preproc, load_preproc, load_glm_params

# parent dir imports
sys.path.append('..')

from nipy_glm_utils import apply_glm
from datasets_extras import fetch_openfmri
from external.nilearn.datasets import _fetch_file

FULL_ID = 'ds000011'
SHORT_ID = 'ds011'
NAME = 'Classification learning and tone-counting'
DESCRIPTION = """
Fourteen participants were trained on two different classification problems
while they were scanned by using fMRI. Participants were trained on one
problem under single-task (ST) conditions and on the other problem while
performing a concurrent tone-counting task. During training, subjects
learned the categories based on trial-by-trial feedback. After training,
subjects received an additional block of probe trials using a mixed
event-related (ER) fMRI paradigm, during which they classified items
that had been trained under either ST or dual-task (DT) conditions. To
measure how well participants had learned under each condition, no feedback
was presented during the probe block, and all items were presented under ST
conditions.  An additional tone-counting localizer scan presented blocks of
the tone counting task (followed by a probe at the end of each block)
compared to rest.

Get full description <a href="https://openfmri.org/dataset/ds000011">\
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

    # this dataset does not contain contrast definitions
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
