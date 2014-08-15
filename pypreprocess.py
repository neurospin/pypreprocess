"""
Synopsis: Run pypreprocess using dataset-specific configuration file given
at command line.
Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import sys
import matplotlib
matplotlib.use('Agg')
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc

if __name__ == "__main__":
    # sanitize command-line usage
    if len(sys.argv) < 2:
        print "\r\nUsage: python %s </path/to/preproc/job.ini>\r\n" % (
            sys.argv[0])
        print ("Example:\r\npython %s scripts/HCP_tfMRI_MOTOR_preproc"
               ".ini\r\n") % sys.argv[0]
        sys.exit(1)

    # consume config file and run pypreprocess back-end
    subjects = do_subjects_preproc(sys.argv[1])

    from pypreprocess.glm_utils import do_subject_glm
    map(do_subject_glm, subjects)
