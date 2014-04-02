import sys
from openfmri import get_options
from openfmri import process_dataset

options = get_options(sys.argv)

process_dataset(
    dataset_id=options.dataset_id, model_id=options.model_id,
    dataset_dir=options.dataset_dir,
    preproc_dir=options.preproc_dir,
    force_download=options.force_download,
    subject_id=options.subject_id,
    skip_preprocessing=options.skip_preprocessing,
    verbose=options.verbose)
