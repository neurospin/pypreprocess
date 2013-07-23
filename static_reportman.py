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

# sanitize command-line
if len(sys.argv) < 2:
    print (
        "\r\nUsage: python %s [OPTIONS] <path_to_dataset_dir>\r\n"
        "\r\nExamples:"
        "\r\n\r\npython static_reportman.py examples/spm_auditory_runs"
        "\r\n") % sys.argv[0]
    sys.exit(1)

# configure option parser
parser = OptionParser()
parser.add_option('--replace-in-path',
                  dest='replaceinpath',
                  default="",
                  help="""specify a token to replace in paths"""
                  )

parser.add_option('--dataset-id',
                  dest='datasetid',
                  default="UNSPECIFIED!",
                  help="""specify id (i.e short description) of dataset"""
                  )

parser.add_option('--subject-preproc-data-json-filename-wildcat',
                  dest='subjectpreprocdatajsonfilenamewildcat',
                  default="sub*/infos.json",
                  help="""specify filename wildcat for json files containing
subject preprocessed data"""
                  )

# parse args and opts
options, args = parser.parse_args()
dataset_dir = args[0]
subject_json_file_glob = os.path.join(
    dataset_dir,
    options.subjectpreprocdatajsonfilenamewildcat)
subject_preproc_data = glob.glob(subject_json_file_glob)

# sanitize
print (
    "Globing subject json file: %s" % subject_json_file_glob)
if not subject_preproc_data:
    raise Warning("No subject json file found!")

# generate reports proper
generate_dataset_preproc_report(
    subject_preproc_data,
    output_dir=dataset_dir,
    dataset_id=options.datasetid,
    replace_in_path=options.replaceinpath.split(','),
    )
