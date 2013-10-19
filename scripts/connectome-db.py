"""
:Author: dohmatob elvis dopgima <gmdopp@gmail.com>

"""

import os
import sys
import glob
import commands
import nibabel
import numpy as np
import joblib
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                   SubjectData
                                                   )

# constants
SESSIONS = ['tfMRI_LANGUAGE', 'tfMRI_MOTOR']
DIRECTIONS = ['LR', 'RL']
FSLMERGE_CMDLINE = "fsl5.0-fslmerge -t %s %s %s"
ACQ_PARAMS = [[-1, 0, 0, 1]] * 3 + [[1, 0, 0, 1]] * 3
TOPUP_CMDLINE = """fsl5.0-topup --verbose --config=b02b0.cnf \
--imain=%s --datain=%s --out=%s/b0_dc_out --fout=%s/b0_dc_fout \
--iout=%s/b0_dc_iout\
"""
APPLYTOPUP_CMDLINE = """fsl5.0-applytopup --imain=%s,%s \
--datain=%s --interp=spline --verbose \
--inindex=1,4 --topup=%s/b0_dc_out --out=%s\
"""

# viviana doesn't want me to normalize
VIVI_IS_HERE = False

# main code follows
if __name__ == '__main__':
    # sanitize input
    if len(sys.argv) < 3:
        print ("\r\nUsage: python %s "
               "<connection_db_data_dir> <output_dir>") % sys.argv[0]
        sys.exit(-1)

    data_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    mem = joblib.Memory(cachedir=os.path.join(output_dir, "cache_dir"),
                        verbose=100)

    def _cached(f):
        """
        Runs a function in a cache 'sandbox'

        """

        return mem.cache(f)

    skipped_subjects = []

    def _skip_subject(subject_id, reason):
        """
        Skips a subject.

        """

        skipped_subjects.append((subject_id, reason))
        print "Skipping subject %s: %s" % (subject_id, reason)

    def _do_b0_dc(subject_id, subject_data_dir):
        """
        Uses FSL's topup tool to correct for phase encoding direction
        bias in HCP subject fMRI data.

        """

        # instantiate subject data object
        subject_data = SubjectData()

        # glob anat
        try:
            subject_data.anat = glob.glob(os.path.join(
                    subject_data_dir,
                    "Structural/T1*_acpc_dc_restore_brain.nii.gz"))[0]
            subject_data.anat = subject_data.anat.replace(".gz", "")
            if not os.path.exists(subject_data.anat):
                nibabel.save(nibabel.load(subject_data.anat + ".gz"),
                         subject_data.anat)
        except IndexError:
            _skip_subject(subject_id, "missing anat data")
            return

        # set id params
        subject_data.subject_id = subject_id
        subject_data.session_id = SESSIONS
        subject_data.output_dir = os.path.join(output_dir, subject_id)

        # prepare func data (b0 distortion correction, etc.)
        subject_data.func = []
        subject_data.sbref = []
        for session in SESSIONS:
            _tmp = os.path.join(subject_data_dir,
                                session, "unprocessed", "3T")

            # fieldmaps for different PEs (Phase Encodings)
            fieldmap_LR = os.path.join(
                _tmp, "%s_LR/%s_3T_SpinEchoFieldMap_LR.nii.gz" % (
                    session, subject_id))
            fieldmap_RL = os.path.join(
                _tmp, "%s_RL/%s_3T_SpinEchoFieldMap_RL.nii.gz" % (
                    session, subject_id))

            # SBRef's for different PEs
            sbref_LR = os.path.join(_tmp,
                                    "%s_LR/%s_3T_%s_LR_SBRef.nii.gz" % (
                    session, subject_id, session))
            sbref_RL = os.path.join(_tmp,
                                    "%s_RL/%s_3T_%s_RL_SBRef.nii.gz" % (
                    session, subject_id, session))

            # raw BOLDs for different PEs
            bold_LR = os.path.join(_tmp,
                                   "%s_LR/%s_3T_%s_LR.nii.gz" % (
                    session, subject_id, session))
            bold_RL = os.path.join(_tmp,
                                   "%s_RL/%s_3T_%s_RL.nii.gz" % (
                    session, subject_id, session))

            for item in [fieldmap_LR,
                         fieldmap_RL,
                         sbref_LR,
                         sbref_RL,
                         bold_LR,
                         bold_RL]:
                if not os.path.exists(item):
                    _skip_subject(subject_id, "Missing file: %s" % item)
                    return

            # save acquisition params
            acq_params_filename = os.path.join(_tmp,
                                               "b0_acquisition_params.txt")
            np.savetxt(acq_params_filename, ACQ_PARAMS, fmt='%i')

            # merged fieldmap output filename
            fieldmap = os.path.join(_tmp, os.path.basename(
                    fieldmap_LR).split("_LR")[0] + ".nii.gz")

            # unwarped BOLD
            unwarped_bold = os.path.join(_tmp,
                                         "%s_3T_%s_unwarped.nii.gz" % (
                    session, subject_id))

            # unwarped SBRef
            unwarped_sbref = os.path.join(
                _tmp, "%s_3T_%s_SBRef_unwarped.nii.gz" % (
                    session, subject_id))

            if not os.path.exists(unwarped_bold):

                # fslmerge
                print FSLMERGE_CMDLINE % (fieldmap, fieldmap_LR, fieldmap_RL)
                print commands.getoutput(FSLMERGE_CMDLINE % (
                        fieldmap, fieldmap_LR, fieldmap_RL))

                # topup
                print TOPUP_CMDLINE % (fieldmap, acq_params_filename,
                                       _tmp, _tmp, _tmp)
                print commands.getoutput(TOPUP_CMDLINE % (
                        fieldmap, acq_params_filename, _tmp, _tmp, _tmp))

                # appytopup
                for lr, rl, unwarped in zip([bold_LR, sbref_LR],
                                            [bold_RL, sbref_RL],
                                            [unwarped_bold, unwarped_sbref]
                                            ):
                    print APPLYTOPUP_CMDLINE % (lr, rl, acq_params_filename,
                                                _tmp, unwarped)
                    print commands.getoutput(APPLYTOPUP_CMDLINE % (
                            lr, rl, acq_params_filename, _tmp, unwarped))

            # do ".nii.gz -> .nii" conversion for SPM
            unwarped_bold = unwarped_bold.replace(".gz", "")
            if not os.path.exists(unwarped_bold):
                nibabel.save(nibabel.load(unwarped_bold + ".gz"),
                             unwarped_bold)

            subject_data.func.append(unwarped_bold)
            subject_data.sbref.append(unwarped_sbref)

        return subject_data

    def subject_factory(n_jobs=-1):
        """
        Factory for subject data.

        """

        subject_stuff = sorted([(x, os.path.basename(x))
                                for x in glob.glob(os.path.join(
                        data_dir, "*")) if os.path.isdir(x)])

        for subject_data in joblib.Parallel(n_jobs=n_jobs, verbose=100)(
            joblib.delayed(_do_b0_dc)(subject_id,
                                      subject_data_dir)
            for subject_data_dir, subject_id in subject_stuff):
            if not subject_data is None:
                yield subject_data

    # preprocess the subjects proper
    n_jobs = int(os.environ['N_JOBS']) if 'N_JOBS' in os.environ else -1
    do_normalize = do_segment = not VIVI_IS_HERE
    preproc_results = do_subjects_preproc(
        subject_factory(n_jobs=n_jobs),
        n_jobs=n_jobs,
        output_dir=output_dir,
        dataset_id='CONNECTOME-DB',
        prepreproc_undergone=(
            "Data was collected with reversed phase-encode "
            "blips (LR/RL), resulting in pairs of images with "
            "distortions going in opposite directions. From "
            "these pairs the susceptibility-induced "
            "off-resonance field was estimated using a "
            "method similar to that described in [Andersson"
            " 2003] as implemented in FSL topup tool [Smith 2004] "
            "and the two images were combined into a single "
            "corrected one."),
        func_to_anat=True,
        do_segment=do_segment,
        do_normalize=do_segment,
        )

    if len(skipped_subjects) > 0:
        print "\r\nSkipped subjects:"
        print "\t     ==================="
        print "\t% 15s | %s" % ("subject id", "reason")
        print "\t     ==================="
        for ss in skipped_subjects:
            print "\t% 15s | %s" % ss
        print
