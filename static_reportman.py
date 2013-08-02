"""
:Module: static_reportman.py
:Synopsis: Generate post-preproc, etc., QA reports of line for a given dataset
:Author: dohmatob elvis dopgima

ABIDE use-case example
----------------------
edohmato@is150118:~/CODE/FORKED/pypreprocess/reporting for j in \
$(ls /vaporific/edohmato/pypreprocess_runs/abide/); do echo;
echo "Generating QA for $j"; echo; python static_reportman.py \
/vaporific/edohmato/pypreprocess_runs/abide/$j "$j_*/infos_DARTEL.json" $j;\
done

"""

import sys
from reporting.preproc_reporter import generate_dataset_preproc_report
import os
import glob
from optparse import OptionParser

# brag!
print "\r\n\t\t +++static-report-man+++\r\n"

# configure option parser
parser = OptionParser()
parser.add_option('--replace-in-path',
                  dest='replace_in_path',
                  default="",
                  help="""specify a token to replace in paths"""
                  )
parser.add_option('--dataset-id',
                  dest='dataset_id',
                  default="UNSPECIFIED!",
                  help="""specify id (i.e short description) of dataset"""
                  )
parser.add_option('--output_dir',
                  dest='output_dir',
                  default=None,
                  help="""specify output directory"""
                  )
parser.add_option('--subject-preproc-data-json-filename-wildcat',
                  dest='subject_preproc_data_json_filename_wildcat',
                  default="sub*/infos.json",
                  help="""specify filename wildcat for json files containing
subject preprocessed data"""
                  )
parser.add_option('--n-jobs',
                  dest='n_jobs',
                  default=1,
                  type=int,
                  help="""number of subprocesses to spawn (defaults to 1)"""
                  )

# parse args and opts
options, args = parser.parse_args()

if len(args) < 1:
    print ("Error: Insufficient number of arguments\nUse the --help"
           " option to get help")
    sys.exit(1)

dataset_dir = args[0]
output_dir = options.output_dir if not options.output_dir is \
    None else dataset_dir

subject_json_file_glob = os.path.join(
    dataset_dir,
    options.subject_preproc_data_json_filename_wildcat)
subject_preproc_data = glob.glob(subject_json_file_glob)

# sanitize
print (
    "Globing subject json file: %s" % subject_json_file_glob)
if not subject_preproc_data:
    raise Warning("No subject json file found!")

# generate reports proper
generate_dataset_preproc_report(
    subject_preproc_data,
    output_dir=output_dir,
    dataset_id=options.dataset_id,
    replace_in_path=options.replace_in_path.split(','),
    n_jobs=options.n_jobs
    )
