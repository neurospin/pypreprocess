"""
Author: DOHMATOB Elvis Dopgima elvis[dot]dohmatob[at]inria[dot]fr
Synopsis: Demo for coregistration in pure python

It demos coregistration on a variety of datasets including:
SPM single-subject auditory, NYU rest, ABIDE, etc.
"""

import os
import glob
import matplotlib.pyplot as plt
from pypreprocess.datasets import fetch_spm_auditory
from nilearn.datasets import fetch_nyu_rest
from pypreprocess.reporting.check_preprocessing import plot_registration
from pypreprocess.coreg import Coregister
from joblib import Memory

# misc
mem = Memory("demos_cache")


def _run_demo(func, anat):
    # fit
    coreg = Coregister().fit(anat, func)

    # apply coreg
    VFk = coreg.transform(func)

    # QA
    plot_registration(anat, VFk, title="before coreg")
    plot_registration(VFk, anat, title="after coreg")
    plt.show()


def _spm_auditory_factory():
    sd = fetch_spm_auditory()
    return sd.func[0], sd.anat

def _nyu_rest_factory(session=1):
    from pypreprocess.nipype_preproc_spm_utils import SubjectData

    nyu_data = fetch_nyu_rest(sessions=[session], n_subjects=7)

    session_func = [x for x in nyu_data.func if "session%i" % session in x]
    session_anat = [
        x for x in nyu_data.anat_skull if "session%i" % session in x]

    for subject_id in set([os.path.basename(os.path.dirname
                                            (os.path.dirname(x)))
                           for x in session_func]):
        # instantiate subject_data object
        subject_data = SubjectData()
        subject_data.subject_id = subject_id
        subject_data.session_id = session

        # set func
        subject_data.func = [x for x in session_func if subject_id in x]
        assert len(subject_data.func) == 1
        subject_data.func = subject_data.func[0]

        # set anat
        subject_data.anat = [x for x in session_anat if subject_id in x]
        assert len(subject_data.anat) == 1
        subject_data.anat = subject_data.anat[0]

        # set subject output directory
        subject_data.output_dir = "/tmp/%s" % subject_id

        subject_data.sanitize(deleteorient=False, niigz2nii=False)

        yield (subject_data.subject_id, subject_data.func[0],
               subject_data.anat)

# spm auditory demo
mem.cache(_run_demo)(*_spm_auditory_factory())

# NYU rest demo
for subject_id, func, anat in _nyu_rest_factory():
    print("%s +++NYU rest %s+++\r\n" % ("\t" * 5, subject_id))
    mem.cache(_run_demo)(func, anat)
