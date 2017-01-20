"""
Author: yannick schwartz, DOHMATOB Elvis Dopgima
Synopsis: Command line interface to preprocess OpenfMRI
"""

import os
import sys

from optparse import OptionParser

from pypreprocess.openfmri import preproc_dataset


parser = OptionParser(usage=(
    '%prog [input_dir] [output_dir]\n\n'
    'Examples:\n\r'
    'Assuming you current directory is .../pypreprocess/examples'
    'python openfmri_preproc.py /tmp/ds001 /tmp/ds001_preproc'
    'python openfmri_preproc.py /tmp/ds001 /tmp/ds001_preproc -s sub001 -O\n'
    'python openfmri_preproc.py /tmp/ds001 /tmp/ds001_preproc -O -D -n 6'))

parser.description = (
    '`input_dir` is the path to an existing '
    'OpenfMRI dataset or where to download it. '
    'The directory name must match a valid OpenfMRI dataset id, '
    'and therefore look like /path/to/dir/{dataset_id}. OpenfMRI datasets '
    'identifiers may be found here: https://openfmri.org/data-sets but '
    'contain only 3 digits. e.g., the valid id for ds000001 is ds001.')

parser.add_option(
    '-s', '--subjects', dest='subjects',
    help=('Process a single subject matching the given id. '
          'A file path may be given, and must contain '
          'a subject_id per line.'))

parser.add_option(
    '-O', '--delete-orient', dest='delete_orient',
    default=False, action="store_true",
    help=('Delete orientation information in nifti files.'))

parser.add_option(
    '-D', '--dartel', dest='dartel',
    default=False, action="store_true",
    help=('Use dartel.'))

parser.add_option(
    '-n', '--n-jobs', dest='n_jobs', type='int',
    default=os.environ.get('N_JOBS', '1'),
    help='Number of parallel jobs.')

options, args = parser.parse_args(sys.argv)
if len(args) < 3:
    options, args = parser.parse_args(sys.argv + ['-h'])
input_dir, output_dir = args[1:]
input_dir = input_dir.rstrip('/')
output_dir = output_dir.rstrip('/')
_, dataset_id = os.path.split(input_dir)

if not dataset_id.startswith('ds') and not os.path.exists(input_dir):
    parser.error("The directory does not exist and "
                 "does not seem to be an OpenfMRI dataset.")

if options.subjects is not None and os.path.exists(options.subjects):
    with open(options.subjects, 'rb') as f:
        restrict = f.read().split()
else:
    restrict = None if options.subjects is None else [options.subjects]

preproc_dataset(data_dir=input_dir,
                output_dir=output_dir,
                restrict_subjects=restrict,
                dartel=options.dartel,
                delete_orient=options.delete_orient,
                n_jobs=options.n_jobs)

print("\r\nAll output written to %s" % output_dir)
