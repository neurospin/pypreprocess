"""
:Module: single_subject_pipeline
:Synopsis: intra-subject (single-subject) preprocessing
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

import sys
import os
import inspect
import time
import numpy as np
import nibabel
import joblib
from collections import namedtuple
import tempfile

# pypreprocess root dir
PYPREPROCESS_DIR = os.path.dirname(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(PYPREPROCESS_DIR)

import reporting.preproc_reporter as preproc_reporter
import reporting.base_reporter as base_reporter
from coreutils.io_utils import (get_basenames,
                                save_vols,
                                save_vol,
                                load_specific_vol,
                                load_vol,
                                load_4D_img,
                                is_niimg,
                                niimg2ndarrays
                                )
from algorithms.slice_timing.spm_slice_timing import fMRISTC
from algorithms.registration.spm_realign import MRIMotionCorrection
from algorithms.registration.affine_transformations import spm_matrix
from algorithms.registration.kernel_smooth import smooth_image
from algorithms.registration.spm_coreg import SPMCoreg
from reporting.base_reporter import (ProgressReport,
                                     PYPREPROCESS_URL
                                     )


# output image prefices
PREPROC_OUTPUT_IMAGE_PRERICES = {'STC': 'A',
                                 'MC': 'R',
                                 'coreg': 'Coreg',
                                 'smoothing': 'S'
                                 }


def do_subject_preproc(subject_data,
                       verbose=True,
                       do_caching=True,
                       do_stc=True,
                       interleaved=False,
                       slice_order='ascending',
                       do_realign=True,
                       do_coreg=True,
                       coreg_func_to_anat=True,
                       fwhm=None,
                       write_preproc_output_images=False,
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

    write_preproc_output_images: bool, optional (default False)
        if set, then the preprocessed images will be written unto disk
        at the end of the pipeline

    Returns
    -------
    dict of preproc output

    """

    # sanitize input args
    for key in ["subject_id",
                "anat",
                "func",
                "n_sessions",
                "output_dir"
                ]:
        assert key in subject_data, "subject_data must have '%s' key" % key

    assert len(subject_data['func']) == subject_data['n_sessions']
    n_sessions = subject_data['n_sessions']

    # print input args
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    print "\r\n"
    for i in args:
        print "\t %s=%s" % (i, values[i])
    print "\r\n"

    # dict of outputs
    output = subject_data.copy()

    # compute basenames of input images
    func_basenames = [get_basenames(func) for func in subject_data['func']]
    anat_basename = get_basenames(subject_data['anat'])

    # create output dir if inexistent
    if not os.path.exists(subject_data['output_dir']):
        os.makedirs(subject_data['output_dir'])

    # prepare for smart caching
    if do_caching:
        mem = joblib.Memory(cachedir=os.path.join(
                subject_data['output_dir'], 'cache_dir'),
                            mmap_mode='r+',
                            verbose=100
                            )

    def _cached(f):
        """
        If caching is enabled, then this wrapper caches calls to a function f.
        Otherwise the behaviour of f is unchanged.

        """

        return mem.cache(f) if do_caching else f

    # prefix for final output images
    func_prefix = ""
    anat_prefix = ""
    if do_stc:
        func_prefix = PREPROC_OUTPUT_IMAGE_PRERICES['STC'] + func_prefix
    if do_realign:
        func_prefix = PREPROC_OUTPUT_IMAGE_PRERICES['MC'] + func_prefix
    if do_coreg:
        if coreg_func_to_anat:
            func_prefix = PREPROC_OUTPUT_IMAGE_PRERICES['coreg'] + func_prefix
        else:
            anat_prefix = PREPROC_OUTPUT_IMAGE_PRERICES['coreg'] + anat_prefix

    if not fwhm is None:
        func_prefix = PREPROC_OUTPUT_IMAGE_PRERICES['smoothing'] + func_prefix

    # cast all images to niimg
    output['func'] = [load_4D_img(x) for x in output['func']]

    if 'anat' in output:
        output['anat'] = load_vol(output['anat'])

    if do_report:
        # generate explanation of preproc steps undergone by subject
        preproc_undergone = preproc_reporter.\
            generate_preproc_undergone_docstring(
            fwhm=fwhm,
            do_slicetiming=do_stc,
            do_realign=do_realign,
            do_coreg=do_coreg,
            )

        # report filenames
        report_log_filename = os.path.join(output['output_dir'],
                                           'report_log.html')
        report_preproc_filename = os.path.join(output['output_dir'],
                                               'report_preproc.html')
        report_filename = os.path.join(output['output_dir'],
                                       'report.html')

        # initialize results gallery
        loader_filename = os.path.join(
            output['output_dir'], "results_loader.php")
        results_gallery = base_reporter.ResultsGallery(
            loader_filename=loader_filename,
            title="Report for subject %s" % output['subject_id'])
        final_thumbnail = base_reporter.Thumbnail()
        final_thumbnail.a = base_reporter.a(href=report_preproc_filename)
        final_thumbnail.img = base_reporter.img()
        final_thumbnail.description = output['subject_id']

        output['results_gallery'] = results_gallery

        # copy web stuff to subject output dir
        base_reporter.copy_web_conf_files(output['output_dir'])

        slice_order = output[
            'slice_order'] if 'slice_order' in output else slice_order
        interleaved = output[
            'interleaved'] if 'interleaved' in output else interleaved

        # initialize progress bar
        subject_progress_logger = base_reporter.ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename,
                                 report_preproc_filename])
        output['progress_logger'] = subject_progress_logger

        # html markup
        preproc = base_reporter.get_subject_report_preproc_html_template(
            ).substitute(
            results=results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            subject_id=output['subject_id'],
            )
        main_html = base_reporter.get_subject_report_html_template(
            ).substitute(
            start_time=time.ctime(),
            subject_id=output['subject_id']
            )

        with open(report_preproc_filename, 'w') as fd:
            fd.write(str(preproc))
            fd.close()
        with open(report_filename, 'w') as fd:
            fd.write(str(main_html))
            fd.close()

    ############################
    # Slice-Timing Correction
    ############################
    if do_stc:
        print "\r\nNODE> Slice-Timing Correction"

        stc_output = []
        original_bold = list(output['func'])
        for sess_func in output['func']:
            sess_func_data, sess_func_affine = niimg2ndarrays(sess_func)
            fmristc = _cached(fMRISTC(slice_order=slice_order,
                                      interleaved=interleaved,
                                      verbose=verbose
                                      ).fit)(raw_data=sess_func_data)

            stc_output.append(nibabel.Nifti1Image(_cached(fmristc.transform)(
                        sess_func_data), sess_func_affine))

        output['func'] = stc_output

        if do_report:
            # generate STC QA thumbs
            preproc_reporter.generate_stc_thumbnails(
                original_bold,
                stc_output,
                output['output_dir'],
                sessions=xrange(n_sessions),
                results_gallery=results_gallery
                )

        # garbage collection
        del fmristc, original_bold

    ######################
    # Motion Correction
    ######################
    if do_realign:
        print "\r\nNODE> Motion Correction"

        mrimc = _cached(MRIMotionCorrection(
                n_sessions=n_sessions, verbose=verbose).fit)(
            [niimg2ndarrays(sess_func)
             for sess_func in output['func']])

        mrimc_output = _cached(mrimc.transform)(reslice=True, concat=True
                                               )

        output['func'] = mrimc_output['realigned_images']
        output['realignment_parameters'] = mrimc_output[
            'realignment_parameters']

        if do_report:
            # generate realignment thumbs
            preproc_reporter.generate_realignment_thumbnails(
                output['realignment_parameters'],
                output['output_dir'],
                sessions=xrange(n_sessions),
                results_gallery=results_gallery
                )

        # garbage collection
        del mrimc

    ###################
    # Coregistration
    ###################
    if do_coreg and 'anat' in output:
        which = "func -> anat" if coreg_func_to_anat else "anat -> func"
        print "\r\nNODE> Coregistration %s" % which

        ref_brain = 'func'
        src_brain = 'anat'
        ref = load_specific_vol(output['func'][0], 0)[0]
        src = output['anat']
        if coreg_func_to_anat:
            ref_brain, src_brain = src_brain, ref_brain
            ref, src = src, ref

        # estimate realignment (affine) params for coreg
        spmcoreg = _cached(SPMCoreg(verbose=verbose).fit)(
            niimg2ndarrays(ref),
            niimg2ndarrays(src))

        # apply coreg
        if coreg_func_to_anat:
            coreg_func = []
            for sess_func in output['func']:
                coreg_func.append(_cached(spmcoreg.transform)(
                        sess_func)['coregistered_source'])
                output['func'] = coreg_func
            src = load_specific_vol(output['func'][0], 0)[0]
        else:
            output['anat'] = _cached(spmcoreg.transform)(
                output['anat'])['coregistered_source']
            src = output['anat']

        if do_report:
            # generate coreg QA thumbs
            preproc_reporter.generate_coregistration_thumbnails(
                (ref, ref_brain),
                (src, src_brain),
                output['output_dir'],
                results_gallery=results_gallery,
                )

        # garbage collection
        del spmcoreg

    ##############
    # Smoothing
    ##############
    if not fwhm is None:
        print ("\r\nNODE> Smoothing with %smm x %smm x %smm Gaussian"
               " kernel") % tuple(fwhm)

        sfunc = []
        for sess_func in output['func']:
            sfunc.append(_cached(smooth_image)(niimg2ndarrays(sess_func),
                                               fwhm))

        output['func'] = sfunc

    if do_report:
        # generate CV thumbs
        preproc_reporter.generate_cv_tc_thumbnail(
            output['func'],
            xrange(n_sessions),
            output['subject_id'],
            output['output_dir'],
            results_gallery=results_gallery)

    # write final output images
    if write_preproc_output_images:
        print "Saving preprocessed images unto disk..."

        # save final func
        func_basenames = func_basenames[0] if (not isinstance(
                func_basenames, basestring) and concat) else func_basenames

        for sess in xrange(n_sessions):
            sess_func = output['func'][sess]
            save_vols(sess_func,
                      output_dir=output['output_dir'],
                      basenames=func_basenames[sess],
                      prefix=func_prefix,
                      concat=concat
                      )

        # save final anat
        save_vol(output['anat'],
                 output_dir=output['output_dir'],
                 basename=anat_basename,
                 prefix=anat_prefix
                 )

        print "...done."

    # finish reporting
    if do_report:
        if shutdown_reloaders:
            base_reporter.ProgressReport().finish(report_preproc_filename)

        print "\r\nHTML report written to %s\r\n" % report_preproc_filename

    return output
