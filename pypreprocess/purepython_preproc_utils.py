"""
:Module: single_subject_preproc_utils
:Synopsis: intra-subject (single-subject) preprocessing in pure python
(no nipype, no SPM, nothing)
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

import os
import inspect
import time
import numpy as np
import nibabel
import inspect
from joblib import Memory
from reporting.preproc_reporter import (
    generate_stc_thumbnails,
    generate_preproc_undergone_docstring
    )
from .io_utils import (get_basenames,
                       save_vols,
                       load_specific_vol,
                       load_4D_img
                       )
from .subject_data import SubjectData
from .slice_timing import fMRISTC
from .realign import MRIMotionCorrection
from .kernel_smooth import smooth_image
from .coreg import Coregister

# output image prefices
PREPROC_OUTPUT_IMAGE_PREFICES = {'STC': 'a',
                                 'MC': 'r',
                                 'coreg': 'c',
                                 'smoothing': 's'
                                 }


def do_subject_preproc(subject_data,
                       verbose=True,
                       do_caching=True,
                       do_stc=False,
                       interleaved=False,
                       slice_order='ascending',
                       do_realign=True,
                       do_coreg=True,
                       coreg_func_to_anat=True,
                       do_cv_tc=True,
                       fwhm=None,
                       write_output_images=2,
                       concat=False,
                       do_report=True,
                       parent_results_gallery=None,
                       shutdown_reloaders=True
                       ):
    """
    API for preprocessing data from single subject (perhaps mutliple sessions)

    Parameters
    ----------
    subject_data: dict
        data from single subject to be preprocessed.
        Obligatory keys and values are:
        n_sessions: int
            number of sessions/runs in the acquisition

        subject_id: string
            subject id

        func: list of n_sessions elements, where each element represents the
        functional data for that session. Functional that for a session can
        be a list of 3D image filenames, a 4D image filename, or a 4D niimg
            functional data

        output_dir: string
           destination directory for all output

    do_caching: bool, optional (default True)
       if set, then `joblib.Memory` will be used to cache costly intermediate
       function calls

    do_stc: bool, optional (default True)
       if set, then Slice-Timing Correction (STC) will be done

    interleaved: bool, optional (default False)
       if set, the it is assumed that the BOLD was acquired in interleaved
       slices

    slice_order: string, optional (default "ascending")
       the acquisition order of the BOLD. This parameter is passed to fMRISTC
       constructor

    do_realign: bool, optional (default True)
        if set, then motion correction will be done

    do_coreg: bool, optional (default True)
        if set, coregistration will be done (to put the BOLD and the structural
        images in register)

    fwhm: list of 3 floats, optional (default None)
        FWHM for smoothing kernel

    write_output_images: bool, optional (default 2)
        Possbile values are:
        0: don't write output images unto disk, return them as niimgs
        1: only write output images corresponding to last stage/node of the
           preprocessing pipeline
        2: write output images after each stage/node of the preprocessing
           pipeline (similar to SPM)

    Returns
    -------
    dict of preproc output

    See also
    ========
    pypreprocess.nipype_preproc_spm_utils

    """

    # # sanitize input args
    # for key in ["output_dir"
    #             ]:
    #     assert key in subject_data, "subject_data must have '%s' key" % key

    dict_input = isinstance(subject_data, dict)

    # print input args
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    print "\r\n"
    for i in args:
        print "\t %s=%s" % (i, values[i])
    print "\r\n"

    if isinstance(subject_data, SubjectData):
        subject_data = subject_data.sanitize()
    else:
        subject_data = SubjectData(**subject_data).sanitize()

    n_sessions = len(subject_data.session_id)
    do_coreg = do_coreg and not subject_data.anat is None

    # compute basenames of input images
    func_basenames = [get_basenames(func) for func in subject_data.func]
    if do_coreg:
        anat_basename = get_basenames(subject_data.anat)

    # prepare for smart caching
    if do_caching:
        mem = Memory(cachedir=os.path.join(
                subject_data.output_dir, 'cache_dir'),
                     verbose=100
                     )

    # prefix for final output images
    func_prefix = ""
    anat_prefix = ""

    # cast all images to niimg
    subject_data.func = [load_4D_img(x) for x in subject_data.func]

    # if 'anat' in output:
    #     subject_data.anat = load_vol(subject_data.anat)

    if do_report:
        # generate explanation of preproc steps undergone by subject
        preproc_undergone = generate_preproc_undergone_docstring(
            fwhm=fwhm,
            slice_timing=do_stc,
            realign=do_realign,
            coregister=do_coreg,
            coreg_func_to_anat=coreg_func_to_anat
            )

        # initialize reports factory
        subject_data.init_report(parent_results_gallery=parent_results_gallery,
                                 preproc_undergone=preproc_undergone,
                                 cv_tc=do_cv_tc)

    ############################
    # Slice-Timing Correction
    ############################
    if do_stc:
        print "\r\nNODE> Slice-Timing Correction"

        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['STC'] + func_prefix

        stc_output = []
        original_bold = subject_data.func
        for sess_func, sess_id in zip(subject_data.func,
                                      xrange(n_sessions)):
            fmristc = mem.cache(fMRISTC(slice_order=slice_order,
                                      interleaved=interleaved,
                                      verbose=verbose
                                      ).fit)(raw_data=sess_func.get_data())

            stc_output.append(mem.cache(fmristc.transform)(
                    sess_func,
                    output_dir=subject_data.output_dir if (
                        write_output_images == 2) else None,
                    basenames=func_basenames[sess_id],
                    prefix=func_prefix))

        subject_data.func = stc_output

        # if do_report:
        #     # generate STC QA thumbs
        #     generate_stc_thumbnails(
        #         original_bold,
        #         stc_output,
        #         subject_data.reports_output_dir,
        #         sessions=xrange(n_sessions),
        #         results_gallery=subject_data.results_gallery
        #         )

        # gc
        del original_bold

        # garbage collection
        del fmristc

    ######################
    # Motion Correction
    ######################
    if do_realign:
        print "\r\nNODE> Motion Correction"

        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['MC'] + func_prefix

        mrimc = mem.cache(MRIMotionCorrection(
                n_sessions=n_sessions, verbose=verbose).fit)(
            [sess_func for sess_func in subject_data.func])

        mrimc_output = mem.cache(mrimc.transform)(
            reslice=True,
            output_dir=subject_data.output_dir if (
                write_output_images == 2) else None,
            prefix=func_prefix,
            basenames=func_basenames
            )

        subject_data.func = mrimc_output['realigned_images']
        subject_data.realignment_parameters = mrimc_output[
            'realignment_parameters']

        # generate realignment thumbs
        if do_report:
            subject_data.generate_realignment_thumbnails()

        # garbage collection
        del mrimc

    ###################
    # Coregistration
    ###################
    if do_coreg and not subject_data.anat is None:
        which = "func -> anat" if coreg_func_to_anat else "anat -> func"
        print "\r\nNODE> Coregistration %s" % which

        if coreg_func_to_anat:
            func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['coreg'] + func_prefix
        else:
            anat_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['coreg'] + anat_prefix

        ref_brain = 'func'
        src_brain = 'anat'
        ref = subject_data.func[0]
        src = subject_data.anat
        if coreg_func_to_anat:
            ref_brain, src_brain = src_brain, ref_brain
            ref, src = src, ref

        # estimate realignment (affine) params for coreg
        coreg = mem.cache(Coregister(verbose=verbose).fit)(ref, src)

        # apply coreg
        if coreg_func_to_anat:
            coreg_func = []
            for sess_func, sess_id in zip(subject_data.func, xrange(
                    n_sessions)):
                coreg_func.append(mem.cache(coreg.transform)(
                        sess_func, output_dir=subject_data.output_dir if (
                            write_output_images == 2) else None,
                        prefix=func_prefix,
                        basenames=func_basenames[sess_id] if coreg_func_to_anat
                        else anat_basename
                        ))
                subject_data.func = coreg_func
            src = load_specific_vol(subject_data.func[0], 0)[0]
        else:
            subject_data.anat = mem.cache(coreg.transform)(
                subject_data.anat)
            src = subject_data.anat

        if do_report:
            # generate coreg QA thumbs
            subject_data.generate_coregistration_thumbnails(
                coreg_func_to_anat=coreg_func_to_anat)

        # garbage collection
        del coreg

    ##############
    # Smoothing
    ##############
    if not fwhm is None:
        print ("\r\nNODE> Smoothing with %smm x %smm x %smm Gaussian"
               " kernel") % tuple(fwhm)

        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['smoothing'] + func_prefix

        sfunc = []
        for sess in xrange(n_sessions):
            sess_func = subject_data.func[sess]

            _tmp = mem.cache(smooth_image)(sess_func,
                                       fwhm)

            # save smoothed func
            if write_output_images == 2:
                _tmp = mem.cache(save_vols)(
                    _tmp,
                    subject_data.output_dir,
                    basenames=func_basenames[sess],
                    prefix=func_prefix,
                    concat=concat
                    )

            sfunc.append(_tmp)

        subject_data.func = sfunc

    # write final output images
    if write_output_images == 1:
        print "Saving preprocessed images unto disk..."

        # save final func
        func_basenames = func_basenames[0] if (not isinstance(
                func_basenames, basestring) and concat) else func_basenames

        _func = []
        for sess in xrange(n_sessions):
            sess_func = subject_data.func[sess]
            _func.append(mem.cache(save_vols)(
                    sess_func,
                    output_dir=subject_data.output_dir,
                    basenames=func_basenames[sess],
                    prefix=func_prefix,
                    concat=concat
                    ))
        subject_data.func = _func

    # finalize
    subject_data.finalize_report(last_stage=shutdown_reloaders)
    subject_data.hardlink_output_files(final=True)

    return subject_data.__dict__ if dict_input else subject_data
