"""
:Module: static_reportman.py
:Synopsis: Generate post-preproc, etc., QA reports of line for a given dataset
:Author: dohmatob elvis dopgima

ABIDE use-case example
----------------------
edohmato@is150118:~/CODE/FORKED/pypreprocess for j in \
$(ls /vaporific/edohmato/pypreprocess_runs/abide/); do echo;
echo "Generating QA for $j"; echo; python static_reportman.py \
/vaporific/edohmato/pypreprocess_runs/abide/$j "$j_*/infos_DARTEL.json" $j;\
done

"""

import sys
from reporting.reporter import generate_dataset_preproc_report
import os
import glob


if len(sys.argv) < 2:
    print ("\r\nUsage: python %s <path_to_dataset_dir> "
           "[subject_preproc_data_json_filename_wildcat] [dataset_id]"
           ) % sys.argv[0]
    sys.exit(1)

dataset_dir = sys.argv[1]
subject_preproc_data_json_filename_wildcat = "sub*/infos.json"

if len(sys.argv) > 2:
    subject_preproc_data_json_filename_wildcat = sys.argv[2]

dataset_id = 'UNSPECIFIED!'
if len(sys.argv) > 3:
    dataset_id = sys.argv[3]

print subject_preproc_data_json_filename_wildcat
subject_preproc_data = glob.glob(os.path.join(
    dataset_dir,
    subject_preproc_data_json_filename_wildcat))

print subject_preproc_data
generate_dataset_preproc_report(
    subject_preproc_data,
    output_dir=dataset_dir,
    dataset_id=dataset_id,
    )
