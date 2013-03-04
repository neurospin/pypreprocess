import sys
from reporting.reporter import generate_dataset_preproc_report
import os
import glob


if len(sys.argv) < 2:
    print ("\r\nUsage: python %s <path_to_dataset_dir> "
    "[subject_preproc_data_json_filename_wildcat] [dataset_id]") % sys.argv[0]
    sys.exit(1)

dataset_dir = sys.argv[1]
subject_preproc_data_json_filename_wildcat = "sub*/infos.json"
if len(sys.argv) > 2:
    subject_preproc_json_filename_wildcart = sys.argv[2]

dataset_id = 'UNSPECIFIED!'
if len(sys.argv) > 3:
    dataset_id = sys.argv[3]
generate_dataset_preproc_report(
    glob.glob(os.path.join(dataset_dir,
                           subject_preproc_data_json_filename_wildcat)),
    output_dir=dataset_dir,
    dataset_id=dataset_id,
    )
