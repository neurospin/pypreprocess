import sys
import os
import numpy as np
import nibabel
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.glm import FMRILinearModel
import pylab as pl
from collections import namedtuple
import joblib

# pypreprocess root dir
PYPREPROCESS_DIR = os.path.dirname(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(PYPREPROCESS_DIR)

# import reporting plugins
import reporting.glm_reporter as glm_reporter

# import tools for preproc
from coreutils.io_utils import three_to_four
from algorithms.slice_timing.spm_slice_timing import fMRISTC
from algorithms.registration.spm_realign import MRIMotionCorrection
from algorithms.registration.kernel_smooth import smooth_image

# subject data model
SubjectData = namedtuple("SubjectData", "func anat output_dir")


def do_subject_preproc(subject_data, n_sessions=1, do_stc=True,
                       slice_order='ascending', interleaved=False,
                       ref_slice=0, do_mc=True, fwhm=None):
    """
    API for preprocessing data from single subject (perhaps mutliple sessions)

    Parameters
    ----------
    subject_data: `SubjectData` instance
        data from single subject to be preprocessed
    **kwargs: value-option dict of optional parameters
        Possible keys and values are:
        XXX

    Returns
    -------
    XXX

    Notes
    -----
    XXX

    """

    output = {'func': subject_data['func']}

    # prepare for smart caching
    mem = joblib.Memory(cachedir=os.path.join(subject_data['output_dir'],
                                               'cache_dir'),
                        verbose=100
                        )

    # STC
    if do_stc:
        print "\r\nNODE> Slice-Timing Correction"
        stc_output = []
        for func in output['func']:
            fmristc = mem.cache(fMRISTC(slice_order=slice_order,
                                        interleaved=interleaved,
                                        ).fit)(raw_data=func)

            stc_output.append(mem.cache(fmristc.transform)(
                    output_dir=subject_data['output_dir']))

        output['func'] = stc_output

    # MC
    if do_mc:
        print "\r\nNODE> tMotion Correction"
        mrimc = mem.cache(MRIMotionCorrection(n_sessions=n_sessions).fit)(
            output['func'])

        mrimc_output = mem.cache(mrimc.transform)(
            subject_data['output_dir'],
            reslice=True,
            concat=True,
            )

        output['func'] = mrimc_output['realigned_files']

    # smoothing
    fmri = three_to_four(output['func'][0])
    if not fwhm is None:
        print ("\r\nNODE> Smoothing with %smm x %smm x %smm Gaussian"
               " kernel") % tuple(fwhm)
        fmri = mem.cache(smooth_image)(fmri, fwhm)

    # GLM
    tr = 7.
    n_scans = 96
    _duration = 6
    epoch_duration = _duration * tr
    conditions = ['rest', 'active'] * 8
    duration = epoch_duration * np.ones(len(conditions))
    onset = np.linspace(0, (len(conditions) - 1) * epoch_duration,
                        len(conditions))
    paradigm = BlockParadigm(con_id=conditions, onset=onset, duration=duration)
    hfcut = 2 * 2 * epoch_duration

    # construct design matrix
    frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    drift_model = 'Cosine'
    hrf_model = 'Canonical With Derivative'
    design_matrix = make_dmtx(frametimes,
                              paradigm, hrf_model=hrf_model,
                              drift_model=drift_model, hfcut=hfcut)

    # plot and save design matrix
    ax = design_matrix.show()
    ax.set_position([.05, .25, .9, .65])
    ax.set_title('Design matrix')
    dmat_outfile = os.path.join(subject_data['output_dir'],
                                'design_matrix.png')
    pl.savefig(dmat_outfile, bbox_inches="tight", dpi=200)

    # specify contrasts
    contrasts = {}
    n_columns = len(design_matrix.names)
    for i in xrange(paradigm.n_conditions):
        contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

    # more interesting contrasts"""
    contrasts['active-rest'] = contrasts['active'] - contrasts['rest']

    # fit GLM
    print('\r\nFitting a GLM (this takes time)...')
    fmri_glm = FMRILinearModel(fmri,
                               design_matrix.matrix,
                               mask='compute')

    fmri_glm.fit(do_scaling=True, model='ar1')

    # save computed mask
    mask_path = os.path.join(subject_data['output_dir'], "mask.nii.gz")
    print "Saving mask image %s..." % mask_path
    nibabel.save(fmri_glm.mask, mask_path)

    # compute bg unto which activation will be projected
    if isinstance(output['func'][0], list):
        anat_img = nibabel.load(output['func'][0][0])
    else:
        anat_img = nibabel.load(output['func'][0])

    anat = anat_img.get_data()

    if anat.ndim == 4:
        anat = anat[..., 0]

    anat_affine = anat_img.get_affine()

    print "Computing contrasts..."
    z_maps = {}
    for contrast_id, contrast_val in contrasts.iteritems():
        print "\tcontrast id: %s" % contrast_id
        z_map, t_map, eff_map, var_map = fmri_glm.contrast(
            contrasts[contrast_id],
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True,
            )

        # store stat maps to disk
        for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                                  [z_map, t_map, eff_map, var_map]):
            map_dir = os.path.join(
                subject_data['output_dir'], '%s_maps' % dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(
                map_dir, '%s.nii.gz' % contrast_id)
            nibabel.save(out_map, map_path)

            # collect zmaps for contrasts we're interested in
            if contrast_id == 'active-rest' and dtype == "z":
                z_maps[contrast_id] = map_path

            print "\t\t%s map: %s" % (dtype, map_path)

        print

    # do stats report
    stats_report_filename = os.path.join(subject_data['output_dir'],
                                         "report_stats.html")
    contrasts = dict((contrast_id, contrasts[contrast_id])
                     for contrast_id in z_maps.keys())
    glm_reporter.generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        z_maps,
        fmri_glm.mask,
        design_matrices=[design_matrix],
        subject_id=subject_data['subject_id'],
        anat=anat,
        anat_affine=anat_affine,
        cluster_th=50,  # we're only interested in this 'large' clusters

        # additional ``kwargs`` for more informative report
        paradigm=paradigm.__dict__,
        TR=tr,
        n_scans=n_scans,
        hfcut=hfcut,
        frametimes=frametimes,
        drift_model=drift_model,
        hrf_model=hrf_model,
        )

    print "\r\nStatistic report written to %s\r\n" % stats_report_filename

if __name__ == '__main__':
    from external.nilearn.datasets import fetch_spm_auditory_data
 
    sd = fetch_spm_auditory_data(os.path.join(os.environ['HOME'],
                                              "CODE/datasets/spm_auditory"))
    subject_data = {'subject_id': 'sub001', 'func': [sd.func], 'anat': sd.anat}

    # run pipeline
    for do_stc in [False, True]:
        pipeline_remark = "_with_stc" if do_stc else ""
        for do_mc in [False, True]:
            pipeline_remark += "_with_mc" if do_mc else ""
            subject_data['output_dir'] = os.path.join(
                '/tmp',
                subject_data['subject_id'],
                pipeline_remark)

            print "\t\t\tpipeline: %s" % pipeline_remark
            do_subject_preproc(subject_data,
                               do_stc=do_stc,
                               do_mc=do_mc,
                               fwhm=[4, 4, 4]
                               )
