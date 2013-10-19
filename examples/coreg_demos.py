"""
:Synopsis: Demo script for coreg.py module
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

import os
import glob
import matplotlib.pyplot as plt
from pypreprocess.datasets import fetch_spm_auditory_data
from pypreprocess.reporting.check_preprocessing import plot_registration
from pypreprocess.coreg import Coregister

if __name__ == '__main__':

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
        sd = fetch_spm_auditory_data(os.path.join(
                os.environ['HOME'], "CODE/datasets/spm_auditory"))

        return sd.func[0], sd.anat

    def _abide_factory(institute="KKI"):
        for scans in sorted(glob.glob(
                "/home/elvis/CODE/datasets/ABIDE/%s_*/%s_*/scans" % (
                    institute, institute))):
            subject_id = os.path.basename(os.path.dirname(
                    os.path.dirname(scans)))
            func = os.path.join(scans, "rest/resources/NIfTI/files/rest.nii")
            anat = os.path.join(scans,
                                "anat/resources/NIfTI/files/mprage.nii")

            yield subject_id, func, anat

    # spm auditory demo
    _run_demo(*_spm_auditory_factory())

    # ABIDE demo
    for subject_id, func, anat in _abide_factory():
        print "%s +++%s+++\r\n" % ("\t" * 5, subject_id)
        _run_demo(func, anat)
