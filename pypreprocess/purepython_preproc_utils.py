"""
:Module: single_subject_preproc_utils
:Synopsis: intra-subject (single-subject) preprocessing in pure python
(no nipype, no SPM, nothing)
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

import os
import inspect
from reporting.preproc_reporter import (
    generate_stc_thumbnails,
    generate_preproc_undergone_docstring
    )

from .external.joblib import Memory
from .io_utils import (get_basenames,
                       save_vols,
                       load_specific_vol,
                       load_4D_img,
                       is_niimg
                       )
from .subject_data import SubjectData
from .slice_timing import fMRISTC
from .realign import MRIMotionCorrection
from .kernel_smooth import smooth_image
from .coreg import Coregister

# output image prefices
PREPROC_OUTPUT_IMAGE_PREFICES = {'STC': 'a',
                                 'MC': 'r',
                                 'smoothing': 's'
                                 }


def _do_subject_slice_timing(subject_data, refslice=0,
                             slice_order="ascending", interleaved=False,
                             caching=True, report=True,
                             write_output_images=2, func_prefix=None,
                             func_basenames=None, ext=None):

    if func_prefix is None:
        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['STC']

    if func_basenames is None:
        func_basenames = [get_basenames(func)
                          for func in subject_data.func]

    # prepare for smart caching
    if caching:
        mem = Memory(cachedir=os.path.join(
                subject_data.output_dir, 'cache_dir'),
                     verbose=100)
    runner = lambda handle: mem.cache(handle) if caching else handle

    stc_output = []
    original_bold = subject_data.func
    for sess_func, sess_id in zip(subject_data.func,
                                  xrange(subject_data.n_sessions)):
        fmristc = runner(fMRISTC(slice_order=slice_order,
                                  interleaved=interleaved,
                                  verbose=True
                                  ).fit)(raw_data=sess_func)

        stc_output.append(runner(fmristc.transform)(
                sess_func,
                output_dir=subject_data.tmp_output_dir if (
                    write_output_images > 0) else None,
                basenames=func_basenames[sess_id],
                prefix=func_prefix, ext=ext))

    subject_data.func = stc_output

    if report:
        # generate STC QA thumbs
        generate_stc_thumbnails(
            original_bold,
            stc_output,
            subject_data.reports_output_dir,
            sessions=xrange(subject_data.n_sessions),
            results_gallery=subject_data.results_gallery
            )

    # gc
    del original_bold

    # garbage collection
    del fmristc

    if write_output_images > 1:
        subject_data.hardlink_output_files()

    return subject_data


def _do_subject_realign(subject_data, reslice=True, register_to_mean=False,
                        caching=True, hardlink_output=True,
                        func_basenames=None, write_output_images=2,
                        report=True, func_prefix=None, ext=None):

    if register_to_mean:
        raise NotImplementedError("Feature pending...")

    if func_prefix is None:
        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['MC']

    if func_basenames is None:
        func_basenames = [get_basenames(func)
                          for func in subject_data.func]

    # prepare for smart caching
    if caching:
        mem = Memory(cachedir=os.path.join(
                subject_data.output_dir, 'cache_dir'),
                     verbose=100
                     )
    runner = lambda handle: mem.cache(handle) if caching else handle

    mrimc = runner(MRIMotionCorrection(
            n_sessions=subject_data.n_sessions, verbose=True).fit)(
        [sess_func for sess_func in subject_data.func])

    mrimc_output = runner(mrimc.transform)(
        reslice=reslice,
        output_dir=subject_data.tmp_output_dir if (
            write_output_images == 2) else None,
        prefix=func_prefix,
        basenames=func_basenames,
        ext=ext
        )

    subject_data.func = mrimc_output['realigned_images']
    subject_data.realignment_parameters = mrimc_output[
        'realignment_parameters']

    # generate realignment thumbs
    if report:
        subject_data.generate_realignment_thumbnails(nipype=False)

    # garbage collection
    del mrimc

    if write_output_images > 1:
        subject_data.hardlink_output_files()

    return subject_data


def _do_subject_coregister(
    subject_data, coreg_func_to_anat=True, reslice=False, caching=True,
    ext=False, write_output_images=2, func_basenames=None, func_prefix="",
    anat_basename=None, anat_prefix="", report=True, verbose=True):

    ref_brain = 'func'
    src_brain = 'anat'
    ref = subject_data.func[0]
    src = subject_data.anat
    if coreg_func_to_anat:
        ref_brain, src_brain = src_brain, ref_brain
        ref, src = src, ref

    # prepare for smart caching
    if caching:
        mem = Memory(cachedir=os.path.join(
                subject_data.output_dir, 'cache_dir'),
                     verbose=100
                     )
    runner = lambda handle: mem.cache(handle) if caching else handle

    # estimate realignment (affine) params for coreg
    coreg = runner(Coregister(verbose=verbose).fit)(ref, src)

    # apply coreg
    if coreg_func_to_anat:
        if func_basenames is None:
            func_basenames = [get_basenames(func)
                              for func in subject_data.func]

        coreg_func = []
        for sess_func, sess_id in zip(subject_data.func, xrange(
                subject_data.n_sessions)):
            coreg_func.append(runner(coreg.transform)(
                    sess_func, output_dir=subject_data.tmp_output_dir if (
                        write_output_images == 2) else None,
                    basenames=func_basenames[sess_id] if coreg_func_to_anat
                    else anat_basename, prefix=func_prefix
                    ))
            subject_data.func = coreg_func
        src = load_specific_vol(subject_data.func[0], 0)[0]
    else:
        if anat_basename is None:
            anat_basename = get_basenames(subject_data.anat)

        subject_data.anat = runner(coreg.transform)(
            subject_data.anat, basename=anat_basename,
            output_dir=subject_data.tmp_output_dir if (
                write_output_images == 2) else None, prefix=anat_prefix)
        src = subject_data.anat

    if report:
        # generate coregistration QA thumbs
        subject_data.generate_coregistration_thumbnails(
            coreg_func_to_anat=coreg_func_to_anat, nipype=False)

    # garbage collection
    del coreg

    if write_output_images > 1:
        subject_data.hardlink_output_files()

    return subject_data


def _do_subject_smooth(subject_data, fwhm, func_prefix=None,
                       write_output_images=2, func_basenames=None,
                       concat=False, caching=True, report=True):

    if func_prefix is None:
        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['smoothing']

    if func_basenames is None:
        func_basenames = [get_basenames(func) for func in subject_data.func]

    # prepare for smart caching
    if caching:
        mem = Memory(cachedir=os.path.join(
                subject_data.output_dir, 'cache_dir'), verbose=100)

    sfunc = []
    for sess in xrange(subject_data.n_sessions):
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
    return subject_data


def do_subject_preproc(subject_data,
                       caching=True,
                       stc=False,
                       refslice=0,
                       interleaved=False,
                       slice_order='ascending',
                       realign=True,
                       coregister=True,
                       coreg_func_to_anat=True,
                       cv_tc=True,
                       fwhm=None,
                       write_output_images=2,
                       concat=False,
                       report=True,
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

    caching: bool, optional (default True)
       if set, then `joblib.Memory` will be used to cache costly intermediate
       function calls

    stc: bool, optional (default True)
       if set, then Slice-Timing Correction (STC) will be done

    interleaved: bool, optional (default False)
       if set, the it is assumed that the BOLD was acquired in interleaved
       slices

    slice_order: string, optional (default "ascending")
       the acquisition order of the BOLD. This parameter is passed to fMRISTC
       constructor

    realign: bool, optional (default True)
        if set, then motion correction will be done

    coreg: bool, optional (default True)
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

    if not write_output_images:
        cv_tc = False

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
    coregister = coregister and not subject_data.anat is None

    # compute basenames of input images
    func_basenames = [get_basenames(subject_data.func[sess]) if not is_niimg(
            subject_data.func[sess]) else "%s.nii.gz" % (
            subject_data.session_id[sess])
                      for sess in xrange(subject_data.n_sessions)]
    if coregister:
        anat_basename = get_basenames(subject_data.anat)

    # prepare for smart caching
    if caching:
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

    if report:
        # generate explanation of preproc steps undergone by subject
        preproc_undergone = generate_preproc_undergone_docstring(
            fwhm=fwhm,
            slice_timing=stc,
            realign=realign,
            coregister=coregister,
            coreg_func_to_anat=coreg_func_to_anat
            )

        # initialize reports factory
        subject_data.init_report(parent_results_gallery=parent_results_gallery,
                                 preproc_undergone=preproc_undergone,
                                 cv_tc=cv_tc)

    ############################
    # Slice-Timing Correction
    ############################
    if stc:
        print "\r\nNODE> Slice-Timing Correction"
        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['STC'] + func_prefix
        subject_data = _do_subject_slice_timing(
            subject_data, refslice=refslice,
            slice_order=slice_order, interleaved=interleaved,
            write_output_images=write_output_images,
            func_prefix=func_prefix,
            report=report,
            func_basenames=func_basenames)

    ######################
    # Motion Correction
    ######################
    if realign:
        print "\r\nNODE> Motion Correction"
        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['MC'] + func_prefix
        subject_data = _do_subject_realign(
            subject_data, reslice=False,
            caching=caching, write_output_images=write_output_images,
            func_prefix=func_prefix,
            func_basenames=func_basenames, report=report)

    ###################
    # Coregistration
    ###################
    if coregister and not subject_data.anat is None:
        which = "func -> anat" if coreg_func_to_anat else "anat -> func"
        print "\r\nNODE> Coregistration %s" % which
        subject_data = _do_subject_coregister(
            subject_data, coreg_func_to_anat=coreg_func_to_anat,
            func_basenames=func_basenames, anat_basename=anat_basename,
            write_output_images=write_output_images, caching=caching,
            func_prefix=func_prefix, anat_prefix=anat_prefix,
            report=report)

    ##############
    # Smoothing
    ##############
    if not fwhm is None:
        print ("\r\nNODE> Smoothing with %smm x %smm x %smm Gaussian"
               " kernel") % tuple(fwhm)
        func_prefix = PREPROC_OUTPUT_IMAGE_PREFICES['smoothing'] + func_prefix
        subject_data = _do_subject_smooth(
            subject_data, fwhm, caching=caching, prefix=func_prefix,
            write_output_images=write_output_images,
            func_basenames=func_basenames, concat=concat)

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

    if write_output_images:
        subject_data.hardlink_output_files(final=True)

    return subject_data.__dict__ if dict_input else subject_data
