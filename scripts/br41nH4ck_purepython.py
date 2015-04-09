import os
import re
import numpy as np
import nibabel
from joblib import Parallel, delayed
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs.mask import intersect_masks
from fetch_brainhack_data import get_subject_data_from_disk
from pypreprocess.purepython_preproc_utils import (do_subject_preproc,
                                                   SubjectData)
from pypreprocess.nipype_preproc_spm_utils import (
    do_subject_preproc as nipype_do_subject_preproc)
print nipype_do_subject_preproc
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report
from pypreprocess.reporting.base_reporter import (ProgressReport,
                                                  pretty_time
                                                  )

output_dir = os.path.join(os.getcwd(), "BrainHack_results")


def make_paradigm(filename, **kwargs):
    """
    Constructs design paradigm from run_*_spmdef.txt file

    """

    text = open(filename).read()
    conditions = []
    onsets = []
    durations = []
    for item in re.finditer(
        "(?P<condition>(?:Unfamiliar|Scrambled|Famous))\t+?"
        "(?P<onset>\S+)\t+?(?P<duration>\S+)",
        text):
        conditions.append(item.group("condition"))
        onsets.append(float(item.group("onset")))
        durations.append(float(item.group("duration")))

    return BlockParadigm(con_id=conditions, onset=onsets,
                         duration=durations,
                         amplitude=np.ones(len(conditions)),
                         **kwargs)


def _preprocess_and_analysis_subject(subject_data,
                                     do_normalize=False,
                                     fwhm=0.,
                                     slicer='z',
                                     cut_coords=6,
                                     threshold=3.,
                                     cluster_th=15
                                     ):
    """
    Preprocesses the subject and then fits (mass-univariate) GLM thereupon.

    """

    # sanitize run_ids:
    # Sub14/BOLD/Run_02/fMR09029-0004-00010-000010-01.nii is garbage,

    # for example
    run_ids = range(9)
    if subject_data['subject_id'] == "Sub14":
        run_ids = [0] + range(2, 9)
        subject_data['func'] = [subject_data['func'][0]] + subject_data[
            'func'][2:]
        subject_data['session_id'] = [subject_data['session_id'][0]
                                      ] + subject_data['session_id'][2:]

    # sanitize subject output dir
    if not 'output_dir' in subject_data:
        subject_data['output_dir'] = os.path.join(
            output_dir, subject_data['subject_id'])

    # preprocess the data
    subject_data = do_subject_preproc(SubjectData(**subject_data),
                                      do_realign=True,
                                      do_coreg=True,
                                      do_report=False,
                                      do_tsdiffana=False
                                      )
    assert not subject_data.anat is None

    # norm
    if do_normalize:
        subject_data = nipype_do_subject_preproc(
            subject_data,
            do_realign=False,
            do_coreg=False,
            do_segment=True,
            do_normalize=True,
            func_write_voxel_sizes=[3, 3, 3],
            anat_write_voxel_sizes=[2, 2, 2],
            fwhm=fwhm,
            hardlink_output=False,
            do_report=False
            )

    # chronometry
    stats_start_time = pretty_time()

    # to-be merged lists, one item per run
    paradigms = []
    frametimes_list = []
    design_matrices = []  # one
    list_of_contrast_dicts = []  # one dict per run
    n_scans = []
    for run_id in run_ids:
        _n_scans = len(subject_data.func[run_id])
        n_scans.append(_n_scans)

        # make paradigm
        paradigm = make_paradigm(getattr(subject_data, 'timing')[run_id])

        # make design matrix
        tr = 2.
        drift_model = 'Cosine'
        hrf_model = 'Canonical With Derivative'
        hfcut = 128.
        frametimes = np.linspace(0, (_n_scans - 1) * tr, _n_scans)
        design_matrix = make_dmtx(
            frametimes,
            paradigm, hrf_model=hrf_model,
            drift_model=drift_model, hfcut=hfcut,
            add_regs=np.loadtxt(getattr(subject_data,
                                        'realignment_parameters')[run_id]),
            add_reg_names=[
                'Translation along x axis',
                'Translation along yaxis',
                'Translation along z axis',
                'Rotation along x axis',
                'Rotation along y axis',
                'Rotation along z axis'
                ]
            )

        # import matplotlib.pyplot as plt
        # design_matrix.show()
        # plt.show()

        paradigms.append(paradigm)
        design_matrices.append(design_matrix)
        frametimes_list.append(frametimes)
        n_scans.append(_n_scans)

        # specify contrasts
        contrasts = {}
        n_columns = len(design_matrix.names)
        for i in xrange(paradigm.n_conditions):
            contrasts['%s' % design_matrix.names[2 * i]] = np.eye(
                n_columns)[2 * i]

        # more interesting contrasts"""
        contrasts['Famous-Unfamiliar'] = contrasts[
            'Famous'] - contrasts['Unfamiliar']
        contrasts['Unfamiliar-Famous'] = -contrasts['Famous-Unfamiliar']
        contrasts['Famous-Scrambled'] = contrasts[
            'Famous'] - contrasts['Scrambled']
        contrasts['Scrambled-Famous'] = -contrasts['Famous-Scrambled']
        contrasts['Unfamiliar-Scrambled'] = contrasts[
            'Unfamiliar'] - contrasts['Scrambled']
        contrasts['Scrambled-Unfamiliar'] = -contrasts['Unfamiliar-Scrambled']

        list_of_contrast_dicts.append(contrasts)

    # importat maps
    z_maps = {}
    effects_maps = {}

    # fit GLM
    print('\r\nFitting a GLM (this takes time) ..')
    fmri_glm = FMRILinearModel([nibabel.concat_images(sess_func)
                                for sess_func in subject_data.func],
                               [design_matrix.matrix
                                for design_matrix in design_matrices],
                               mask='compute')
    fmri_glm.fit(do_scaling=True, model='ar1')

    print "... done.\r\n"

    # save computed mask
    mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")

    print "Saving mask image to %s ..." % mask_path
    nibabel.save(fmri_glm.mask, mask_path)
    print "... done.\r\n"

    # replicate contrasts across runs
    contrasts = dict((cid, [contrasts[cid]
                            for contrasts in list_of_contrast_dicts])
                     for cid, cval in contrasts.iteritems())

    # compute effects
    for contrast_id, contrast_val in contrasts.iteritems():
        print "\tcontrast id: %s" % contrast_id
        z_map, eff_map = fmri_glm.contrast(
            contrast_val,
            con_id=contrast_id,
            output_z=True,
            output_stat=False,
            output_effects=True,
            output_variance=False
            )

        # store stat maps to disk
        for map_type, out_map in zip(['z', 'effects'], [z_map, eff_map]):
            map_dir = os.path.join(
                subject_data.output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            print "\t\tWriting %s ..." % map_path
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if map_type == 'z':
                z_maps[contrast_id] = map_path

            if map_type == 'effects':
                effects_maps[contrast_id] = map_path

    # remove repeated contrasts
    contrasts = dict((cid, cval[0]) for cid, cval in contrasts.iteritems())

    # do stats report
    stats_report_filename = os.path.join(getattr(subject_data,
                                                 'reports_output_dir',
                                                 subject_data.output_dir),
                                         "report_stats.html")
    generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        z_maps,
        fmri_glm.mask,
        threshold=threshold,
        cluster_th=cluster_th,
        slicer=slicer,
        cut_coords=cut_coords,
        anat=nibabel.load(subject_data.anat).get_data(),
        anat_affine=nibabel.load(subject_data.anat).get_affine(),
        design_matrices=design_matrices,
        subject_id=subject_data.subject_id,
        start_time=stats_start_time,
        title="GLM for subject %s" % subject_data.subject_id,

        # additional ``kwargs`` for more informative report
        TR=tr,
        n_scans=n_scans,
        hfcut=hfcut,
        drift_model=drift_model,
        hrf_model=hrf_model,
        paradigm=dict(("Run_%02i" % (run_id + 1), paradigms[run_id].__dict__)
                      for run_id in run_ids),
        frametimes=dict(("Run_%02i" % (run_id + 1), frametimes_list[run_id])
                        for run_id in run_ids),
        # fwhm=fwhm
        )

    ProgressReport().finish_dir(subject_data.output_dir)
    print "\r\nStatistic report written to %s\r\n" % stats_report_filename

    return contrasts, effects_maps, z_maps, mask_path


if __name__ == '__main__':
    # global variables
    slicer = 'z'
    cut_coords = 5
    threshold = 3.
    cluster_th = 15

    # run intra-subject GLM (one per subject) and the collect the results
    # to form input for group-level analysis
    n_jobs = int(os.environ.get('N_JOBS', -1))
    group_glm_inputs = Parallel(n_jobs=n_jobs, verbose=100)(delayed(
            _preprocess_and_analysis_subject)(
            get_subject_data_from_disk("Sub%02i" % (subject_id + 1)),
            do_normalize=True,
            fwhm=[8., 8., 8.],
            threshold=threshold,
            slicer=slicer,
            cut_coords=cut_coords,
            cluster_th=cluster_th
            ) for subject_id in range(
            16))

    # chronometry
    stats_start_time = pretty_time()

    # compute group mask
    print "\r\nComputing group mask ..."
    mask_images = [subject_glm_results[3]
                   for subject_glm_results in group_glm_inputs]
    group_mask = nibabel.Nifti1Image(intersect_masks(mask_images
                                                   ).astype(np.uint8),
                                   nibabel.load(mask_images[0]
                                                ).get_affine())
    print "... done.\r\n"
    print "Group GLM"
    contrasts = [
        subject_glm_results
        for subject_glm_results in group_glm_inputs]
    contrasts = group_glm_inputs[0][0]
    sujects_effects_maps = [subject_glm_results[1]
                           for subject_glm_results in group_glm_inputs]
    group_level_z_maps = {}
    design_matrix = np.ones(len(sujects_effects_maps)
                            )[:, np.newaxis]  # only the intercept
    for contrast_id in contrasts:
        print "\tcontrast id: %s" % contrast_id

        # effects maps will be the input to the second level GLM
        first_level_image = nibabel.concat_images(
            [x[contrast_id] for x in sujects_effects_maps])

        # fit 2nd level GLM for given contrast
        group_model = FMRILinearModel(first_level_image,
                                    design_matrix, group_mask)
        group_model.fit(do_scaling=False, model='ols')

        # specify and estimate the contrast
        contrast_val = np.array(([[1.]]))  # the only possible contrast !
        z_map, = group_model.contrast(contrast_val,
                                    con_id='one_sample %s' % contrast_id,
                                    output_z=True)

        # save map
        map_dir = os.path.join(output_dir, 'z_maps')
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(map_dir, '2nd_level_%s.nii.gz' % (
                contrast_id))
        print "\t\tWriting %s ..." % map_path
        nibabel.save(z_map, map_path)

        group_level_z_maps[contrast_id] = map_path

    # do stats report
    stats_report_filename = os.path.join(output_dir,
                                         "report_stats.html")
    generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        group_level_z_maps,
        group_mask,
        threshold=threshold,
        cluster_th=cluster_th,
        design_matrices=[design_matrix],
        subject_id="sub001",
        start_time=stats_start_time,
        title='Group GLM for br41nh4ck',
        slicer=slicer,
        cut_coords=cut_coords
        )

    ProgressReport().finish_dir(output_dir)
    print "\r\nStatistic report written to %s\r\n" % stats_report_filename
