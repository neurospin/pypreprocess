import reporting.reporter as reporter
import nipype_preproc_spm_utils
import os
import time
import shutil
import sys
import pylab as pl
from reporting.reporter import generate_subject_preproc_report
import glob

# find package path
root_dir = os.path.split(os.path.abspath(__file__))[0]

func_files = sorted(glob.glob(sys.argv[1]))

anat_file = None
if len(sys.argv) > 2:
    anat_file = sys.argv[2]
rp_files = None

if len(sys.argv) > 3:
    rp_files = sys.argv[3]

generate_subject_preproc_report(
    func_files,
    anat_file=anat_file,
    rp_files=rp_files,
    subject_id="johndoe"
    )
