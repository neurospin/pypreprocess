"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

# standard imports
import os
import sys
import time
import warnings
import inspect
import numpy as np
import nibabel
from matplotlib.pyplot import cm

from slice_timing import get_slice_indices
from conf_parser import _generate_preproc_pipeline

# import joblib API
from joblib import (Parallel,
                    delayed,
                    Memory as JoblibMemory
                    )

# import nipype API
import nipype.interfaces.spm as spm
from nipype.caching import Memory as NipypeMemory
from configure_spm import configure_spm

# import API for i/o
from .io_utils import (load_specific_vol,
                       ravel_filenames,
                       unravel_filenames,
                       get_vox_dims,
                       niigz2nii,
                       resample_img,
                       compute_output_voxel_size,
                       sanitize_fwhm
                       )
from subject_data import SubjectData

# import API for reporting
from .reporting.base_reporter import (
    ResultsGallery,
    ProgressReport,
    copy_web_conf_files,
    get_module_source_code,
    dict_to_html_ul
    )
from .reporting.preproc_reporter import (
    _set_templates,
    generate_preproc_undergone_docstring,
    get_dataset_report_log_html_template,
    get_dataset_report_preproc_html_template,
    get_dataset_report_html_template,
    generate_stc_thumbnails
    )

# import pure python preproc API
from purepython_preproc_utils import (
    _do_subject_slice_timing as _pp_do_subject_slice_timing,
    _do_subject_realign as _pp_do_subject_realign,
    _do_subject_coregister as _pp_do_subject_coregister)

# configure SPM
EPI_TEMPLATE = SPM_DIR = SPM_T1_TEMPLATE = T1_TEMPLATE = None
GM_TEMPLATE = WM_TEMPLATE = CSF_TEMPLATE = None


def _configure_backends(spm_dir=None, matlab_exec=None):
    global SPM_DIR, EPI_TEMPLATE, SPM_T1_TEMPLATE, T1_TEMPLATE
    global GM_TEMPLATE, WM_TEMPLATE, CSF_TEMPLATE

    _ = configure_spm(spm_dir=spm_dir, matlab_exec=matlab_exec)

    if _[0]:
        if os.path.isdir(_[0]):
            SPM_DIR = _[0]

            # configure template images
            EPI_TEMPLATE = os.path.join(SPM_DIR, 'templates/EPI.nii')
            SPM_T1_TEMPLATE = os.path.join(SPM_DIR, "templates/T1.nii")
            T1_TEMPLATE = "/usr/share/data/fsl-mni152-templates/avg152T1.nii"
            if not os.path.isfile(T1_TEMPLATE):
                T1_TEMPLATE += '.gz'
                if not os.path.exists(T1_TEMPLATE):
                    T1_TEMPLATE = SPM_T1_TEMPLATE
            GM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/grey.nii')
            WM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/white.nii')
            CSF_TEMPLATE = os.path.join(SPM_DIR, 'tpm/csf.nii')

            _set_templates(spm_dir=SPM_DIR)

try:
    _configure_backends()
except AssertionError:
    pass


def _do_subject_slice_timing(subject_data, TR, TA=None,
                             refslice=0, slice_order="ascending",
                             interleaved=False, caching=True,
                             report=True, software="spm",
                             hardlink_output=True):
    """
    Slice-Timing Correction.

    Parameters
    ----------
    subject_data: `SubjectData` instance
       object that encapsulates the date for the subject (should have fields
       like func, anat, output_dir, etc.)

    TR: float
        Repetition time for the fMRI acquisition

    TA: float, optional (default None)
       Time of Acaquisition

    """

    if not subject_data.func:
        warnings.warn("subject_data.func=%s (empty); skipping STC!" % (
                subject_data.func))
        return subject_data

    software = software.lower()

    # sanitize subject_data (do things like .nii.gz -> .nii conversion, etc.)
    subject_data.sanitize(niigz2nii=(software == spm))

    # compute nslices
    nslices = load_specific_vol(subject_data.func[0], 0)[0].shape[2]
    assert 1 <= refslice <= nslices, refslice

    # compute slice indices / order
    if not isinstance(slice_order, basestring):
        slice_order = np.array(slice_order) - 1
    slice_order = get_slice_indices(nslices, slice_order=slice_order,
                                    interleaved=interleaved)

    # use pure python (pp) code ?
    if software == "python":
        return _pp_do_subject_slice_timing(subject_data, refslice=refslice,
                                           slice_order=slice_order,
                                           caching=caching, report=report)

    # sanitize software choice
    if software != "spm":
        raise NotImplementedError(
            "Only SPM is supported; got software='%s'" % software)

    # compute TA (possibly from formula specified as a string)
    if isinstance(TA, basestring):
        TA = TA.replace("/", "* 1. /")
        TA = eval(TA)

    # compute time of acqusition
    if TA is None:
        TA = TR * (1. - 1. / nslices)

   # run pipeline
    if caching:
        cache_dir = cache_dir = os.path.join(subject_data.output_dir,
                                             'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        stc = subject_data.mem.cache(spm.SliceTiming)
    else:
        stc = spm.SliceTiming().run

    stc_func = []
    subject_data.nipype_results['slice_timing'] = []
    for sess_func in subject_data.func:
        stc_result = stc(in_files=sess_func, time_repetition=TR,
                         time_acquisition=TA, num_slices=nslices,
                         ref_slice=refslice + 1,
                         slice_order=list(slice_order + 1),  # SPM
                         ignore_exception=False
                         )
        if stc_result.outputs is None:
            subject_data.failed = True
            return subject_data
        else:
            subject_data.nipype_results['slice_timing'].append(stc_result)
            stc_func.append(stc_result.outputs.timecorrected_files)

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    # generate STC QA thumbs
    if report:
        generate_stc_thumbnails(
            subject_data.func,
            stc_func,
            subject_data.reports_output_dir,
            sessions=subject_data.session_id,
            results_gallery=subject_data.results_gallery
            )

    subject_data.func = stc_func

    return subject_data.sanitize()


def _do_subject_realign(subject_data, reslice=False, register_to_mean=False,
                        caching=True, report=True, software="spm",
                        hardlink_output=True, **kwargs):
    """
    Wrapper for running spm.Realign with optional reporting.

    If subject_data has a `results_gallery` attribute, then QA thumbnails will
    be commited after this node is executed

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functional images (subject_data.func) are to be
        realigned.

    reslice: bool, optional (default False)
        if set, then realigned images will be resliced

    register_to_mean: bool, optional (default False)
        if set, then realignment will be to the mean functional image

    software: string, optional (default "spm")
        software to use for realignment; can be "spm", "fsl", or "python

    caching: bool, optional (default True)
        if true, then caching will be enabled

    report: bool, optional (default True)
       flag controlling whether post-preprocessing reports should be generated

    **kwargs:
       additional parameters to the back-end (SPM, FSL, python)

    Returns
    -------
    subject_data: `SubjectData` object
        preprocessed subject_data. The func field (subject_data.func) now
        contain the realigned functional images of the subject

        New Attributes
        ==============
        realignment_parameters: string of list of strings
            filename(s) containing realignment parameters from spm.Realign node

        subject_data.nipype_results['realign']: Nipype output object
            (raw) result of running spm.Realign

    Notes
    -----
    Input subject_data is modified.

    """

    if not subject_data.func:
        warnings.warn("subject_data.func=%s (empty); skipping MC!" % (
                subject_data.func))
        return subject_data

    software = software.lower()

    # sanitize subject_data (do things like .nii.gz -> .nii conversion, etc.)
    subject_data.sanitize(niigz2nii=(software == "spm"))

    # use pure python (pp) code ?
    if software == "python":
        return _pp_do_subject_realign(subject_data, register_to_mean=False,
                                      report=report, caching=caching,
                                      reslice=reslice
                                      )

    if software != "spm":
        raise NotImplementedError("Only SPM is supported; got '%s'" % software)

    # jobtype
    jobtype = "estwrite" if reslice else "estimate"

    # XXX with nipype.caching, jobtype="estimate" modifies not even the input
    # file headers!
    if caching:
        jobtype = "estwrite"

    if not hasattr(subject_data, 'nipype_results'):
        subject_data.nipype_results = {}

    # create node
    if caching:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir)
        realign = subject_data.mem.cache(spm.Realign)
    else:
        realign = spm.Realign().run

    # run node
    realign_result = realign(
        in_files=subject_data.func,
        register_to_mean=register_to_mean,
        jobtype=jobtype, **kwargs
        )

    # failed node
    if realign_result.outputs is None:
        subject_data.failed = True
        return subject_data

    # collect output
    subject_data.func = realign_result.outputs.realigned_files

    subject_data.realignment_parameters = \
        realign_result.outputs.realignment_parameters
    if isinstance(subject_data.realignment_parameters, basestring):
        assert subject_data.n_sessions == 1
        subject_data.realignment_parameters = [
            subject_data.realignment_parameters]

    if register_to_mean and jobtype == "estwrite":
        subject_data.mean_realigned_file = \
            realign_result.outputs.mean_image

    subject_data.nipype_results['realign'] = realign_result

    if isinstance(subject_data.func, basestring):
        assert subject_data.n_sessions == 1
        subject_data.func = [subject_data.func]
    if subject_data.n_sessions == 1 and len(subject_data.func) > 1:
        subject_data.func = [subject_data.func]

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    # generate realignment thumbs
    if report:
        subject_data.generate_realignment_thumbnails()

    return subject_data.sanitize()


def _do_subject_coregister(subject_data, reslice=False,
                           coreg_anat_to_func=False, caching=True,
                           report=True, software="spm", hardlink_output=True,
                           **kwargs):
    """
    Wrapper for running spm.Coregister with optional reporting.

    If subject_data has a `results_gallery` attribute, then QA thumbnails will
    be commited after this node is executed

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functional and anatomical images (subject_data.func
        and subject_data.anat) are to be coregistered

    caching: bool, optional (default True)
        if true, then caching will be enabled

    coreg_anat_to_func: bool, optional (default False)
       if set, then functional data (subject_data.func) will be the reference
       during coregistration. By default the anatomical data
       (subject_data.anat) if the reference, to ensure a precise registration
       (since anatomical data has finer resolution)

    report: bool, optional (default True)
       flag controlling whether post-preprocessing reports should be generated

    **kwargs:
       additional parameters to the back-end (SPM, FSL, python)

    Returns
    -------
    subject_data: `SubjectData` object
        preprocessed subject_data. The func and anatomical fields
        (subject_data.func and subject_data.anat) now contain the oregistered
        and anatomical images functional images of the subject

        New Attributes
        ==============
        subject_data.nipype_results['coregister']: Nipype output object
            (raw) result of running spm.Coregister

    Notes
    -----
    Input subject_data is modified.

    """

    if not subject_data.func:
        warnings.warn(
            "subject_data.func=%s (empty); skipping coregistration!" % (
                subject_data.func))
        return subject_data

    if not subject_data.anat:
        warnings.warn(
            "subject_data.anat=%s (empty); skipping coregistration!" % (
                subject_data.anat))
        return subject_data

    software = software.lower()

    # sanitize subject_data (do things like .nii.gz -> .nii conversion, etc.)
    subject_data.sanitize(niigz2nii=(software == "spm"))

    # use pure python (pp) code ?
    if software == "python":
        return _pp_do_subject_coregister(
            subject_data,
            reslice=reslice,
            coreg_func_to_anat=not coreg_anat_to_func,
            report=report, caching=caching,
            )

    # sanitize software choice
    if not software in ["spm", "fsl"]:
        raise NotImplementedError(
            "Only SPM is supported; got '%s'" % software)

    # sanitize subject_data (do things like .nii.gz -> .nii conversion, etc.)
    subject_data.sanitize(niigz2nii=(software == "spm"))

    # jobtype
    jobtype = "estwrite" if reslice else "estimate"

    cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    joblib_mem = JoblibMemory(cache_dir)

    # create node
    if caching:
        if software == "spm":
            coreg = NipypeMemory(base_dir=cache_dir).cache(spm.Coregister)
        elif software == "python":
            coreg = JoblibMemory(cache_dir)
    else:
        coreg = spm.Coregister().run

    # config node
    apply_to_files = []
    ref_brain = 'anat'
    src_brain = 'func'

    if coreg_anat_to_func:
        ref_brain, src_brain = src_brain, ref_brain
        if hasattr(subject_data, "mean_realigned_file"):
            coreg_target = subject_data.mean_realigned_file
        else:
            coreg_target = subject_data.func[0]
        if subject_data.anat is None:
            if not subject_data.hires is None:
                coreg_source = subject_data.hires
        else:
            coreg_source = subject_data.anat
    else:
        coreg_target = subject_data.anat
        if hasattr(subject_data, "mean_realigned_file"):
            coreg_source = subject_data.mean_realigned_file
        else:
            ref_func = joblib_mem.cache(load_specific_vol)(
                subject_data.func if isinstance(subject_data.func, basestring)
                else subject_data.func[0], 0)[0]
            coreg_source = os.path.join(subject_data.tmp_output_dir,
                                        "_coreg_first_func_vol.nii")

            joblib_mem.cache(nibabel.save)(ref_func, coreg_source)

        apply_to_files, file_types = ravel_filenames(subject_data.func)

    # run node
    coreg_result = coreg(target=coreg_target,
                         source=coreg_source,
                         apply_to_files=apply_to_files,
                         jobtype=jobtype,
                         ignore_exception=False
                         )

    # failed node ?
    if coreg_result.outputs is None:
        subject_data.nipype_results['coreg'] = coreg_result
        subject_data.failed = True
        return subject_data

    # collect output
    subject_data.nipype_results['coreg'] = coreg_result
    if coreg_anat_to_func:
        subject_data.anat = coreg_result.outputs.coregistered_source
    else:
        coregistered_files = coreg_result.outputs.coregistered_files
        if isinstance(coregistered_files, basestring):
            coregistered_files = [coregistered_files]
        subject_data.func = unravel_filenames(
            coregistered_files, file_types)

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    # generate coregistration thumbs
    if report:
        subject_data.generate_coregistration_thumbnails()

    return subject_data.sanitize()


def _do_subject_segment(subject_data, output_modulated_tpms=True,
                        normalize=False, caching=True, report=True,
                        software="spm", hardlink_output=True):
    """
    Wrapper for running spm.Segment with optional reporting.

    If subject_data has a `results_gallery` attribute, then QA thumbnails will
    be commited after this node is executed

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose anatomical image (subject_data.anat) is to be
        segmented

    output_modulated_tpms: bool, optional (default False)
        if set, then modulated TPMS will be produced (alongside unmodulated
        TPMs); this can be useful for VBM

    caching: bool, optional (default True)
        if true, then caching will be enabled

    normalize: bool, optional (default False)
        flag indicating whether warped brain compartments (gm, wm, csf) are to
        be generated (necessary if the caller wishes the brain later)

    report: bool, optional (default True)
       flag controlling whether post-preprocessing reports should be generated

    Returns
    -------
    subject_data: `SubjectData` object
        preprocessed subject_data

        New Attributes
        ==============
        subject_data.nipype_results['segment']: Nipype output object
            (raw) result of running spm.Segment

        subject_data.gm: string
            path to subject's segmented gray matter image in native space

        subject_data.wm: string
            path to subject's segmented white matter image in native space

        subject_data.csf: string
            path to subject's CSF image in native space

        if normalize then the following additional data fiels are
        populated:

        subject_data.wgm: string
            path to subject's segmented gray matter image in standard space

        subject_data.wwm: string
            path to subject's segmented white matter image in standard space

        subject_data.wcsf: string
            path to subject's CSF image in standard space


    Notes
    -----
    Input subject_data is modified.

    """

    # sanitize software choice
    software = software.lower()
    if software != "spm":
        raise NotImplementedError("Only SPM is supported; got '%s'" % software)

    # sanitize subject_data (do things like .nii.gz -> .nii conversion, etc.)
    subject_data.sanitize(niigz2nii=(software == "spm"))

    # prepare for smart caching
    if caching:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        segment = NipypeMemory(base_dir=cache_dir).cache(spm.Segment)
    else:
        segment = spm.Segment().run

    # configure node
    if not normalize:
        gm_output_type = [False, False, True]
        wm_output_type = [False, False, True]
        csf_output_type = [False, False, True]
    else:
        gm_output_type = [output_modulated_tpms, True, True]
        wm_output_type = [output_modulated_tpms, True, True]
        csf_output_type = [output_modulated_tpms, True, True]

    # run node
    segment_result = segment(
        data=subject_data.anat,
        gm_output_type=wm_output_type,
        wm_output_type=gm_output_type,
        csf_output_type=csf_output_type,
        tissue_prob_maps=[GM_TEMPLATE, WM_TEMPLATE, CSF_TEMPLATE],
        ignore_exception=False
        )

    # failed node
    subject_data.nipype_results['segment'] = segment_result
    if segment_result.outputs is None:
        subject_data.failed = True
        return subject_data

    # collect output
    subject_data.parameter_file = segment_result.outputs.transformation_mat
    subject_data.nipype_results['segment'] = segment_result
    subject_data.gm = segment_result.outputs.native_gm_image
    subject_data.wm = segment_result.outputs.native_wm_image
    subject_data.csf = segment_result.outputs.native_csf_image
    if normalize:
        subject_data.mwgm = segment_result.outputs.modulated_gm_image
        subject_data.mwwm = segment_result.outputs.modulated_wm_image
        subject_data.mwcsf = segment_result.outputs.modulated_csf_image

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()

    # generate segmentation thumbs
    if report:
        subject_data.generate_segmentation_thumbnails()

    return subject_data.sanitize()


def _do_subject_normalize(subject_data, fwhm=0., anat_fwhm=0., caching=True,
                          func_write_voxel_sizes=[3, 3, 3],
                          anat_write_voxel_sizes=[1, 1, 1],
                          report=True, software="spm",
                          hardlink_output=True
                          ):
    """
    Wrapper for running spm.Segment with optional reporting.

    If subject_data has a `results_gallery` attribute, then QA thumbnails will
    be commited after this node is executed

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functiona and anatomical images (subject_data.func
        and subject_data.anat) are to be normalized (warped into standard
        spac)

    caching: bool, optional (default True)
        if true, then caching will be enabled

    fwhm: float or list of 3 floats, optional (default 0)
        FWHM for smoothing the functional data (subject_data.func).
        If normalize is set, then this parameter is based to spm.Normalize,
        else spm.Smooth is used to explicitly smooth the functional data.

        If spm.Smooth is used for smoothing (i.e if normalize if False
        and fwhm is not 0), then subject_data.nipype_results['smooth']
        will contain the result from the spm.Smooth node.

    report: bool, optional (default True)
       flag controlling whether post-preprocessing reports should be generated

    Returns
    -------
    subject_data: `SubjectData` object
        preprocessed subject_data. The func and anat fields
        (subject_data.func and subject_data.anat) now contain the normalizeed
        functional and anatomical images of the subject

        New Attributes
        ==============
        subject_data.nipype_results['normalize']: Nipype output object
            (raw) result of running spm.Normalize

    Notes
    -----
    Input subject_data is modified

    """

    # sanitize software choice
    software = software.lower()
    if software != "spm":
        raise NotImplementedError("Only SPM is supported; got '%s'" % software)

    # sanitize subject_data (do things like .nii.gz -> .nii conversion, etc.)
    subject_data.sanitize(niigz2nii=(software == "spm"))

    # prepare for smart caching
    if caching:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        normalize = NipypeMemory(base_dir=cache_dir).cache(spm.Normalize)
    else: normalize = spm.Normalize().run

    segmented = 'segment' in subject_data.nipype_results

    # configure node for normalization
    if not segmented:
        # learn T1 deformation without segmentation
        t1_template = niigz2nii(SPM_T1_TEMPLATE,
                                output_dir=subject_data.output_dir)
        normalize_result = normalize(source=subject_data.anat,
                                     template=t1_template,
                                     write_preserve=False,
                                     )
        parameter_file = normalize_result.outputs.normalization_parameters
    else:
        parameter_file = subject_data.nipype_results[
            'segment'].outputs.transformation_mat

    subject_data.parameter_file = parameter_file

    # do normalization proper
    for brain_name, brain, cmap in zip(
        ['anat', 'func'], [subject_data.anat, subject_data.func],
        [cm.gray, cm.spectral]):
        if not brain: continue
        if segmented:
            if brain_name == 'func':
                apply_to_files, file_types = ravel_filenames(subject_data.func)
                if func_write_voxel_sizes is None:
                    write_voxel_sizes = get_vox_dims(apply_to_files)
                else: write_voxel_sizes = func_write_voxel_sizes
            else:
                apply_to_files = subject_data.anat
                if anat_write_voxel_sizes is None:
                    write_voxel_sizes = get_vox_dims(apply_to_files)
                else: write_voxel_sizes = anat_write_voxel_sizes
                apply_to_files = subject_data.anat

            # run node
            normalize_result = normalize(
                parameter_file=parameter_file,
                apply_to_files=apply_to_files,
                write_voxel_sizes=tuple(write_voxel_sizes),
                # write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_interp=1,
                jobtype='write',
                ignore_exception=False
                )

            # failed node ?
            if normalize_result.outputs is None:
                subject_data.nipype_results['normalize_%s' % brain_name
                                            ] = normalize_result
                subject_data.failed = True
                return subject_data
        else:
            if brain_name == 'func':
                apply_to_files, file_types = ravel_filenames(
                    subject_data.func)
                apply_to_files, file_types = ravel_filenames(subject_data.func)
                if func_write_voxel_sizes is None:
                    write_voxel_sizes = get_vox_dims(apply_to_files)
                else: write_voxel_sizes = func_write_voxel_sizes
            else:
                apply_to_files = subject_data.anat
                apply_to_files, file_types = ravel_filenames(subject_data.func)
                if anat_write_voxel_sizes is None:
                    write_voxel_sizes = get_vox_dims(apply_to_files)
                else: write_voxel_sizes = anat_write_voxel_sizes

            # run node
            normalize_result = normalize(
                parameter_file=parameter_file,
                apply_to_files=apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=tuple(write_voxel_sizes),
                write_wrap=[0, 0, 0],
                write_interp=1,
                jobtype='write',
                ignore_exception=False
                )

            # failed node
            subject_data.nipype_results['normalize_%s' % brain_name
                                        ] = normalize_result
            if normalize_result is None:
                # catch dirty exception from SPM back-end
                subject_data.failed = True
                return subject_data

            if brain_name == 'func':
                subject_data.func = unravel_filenames(
                    normalize_result.outputs.normalized_files, file_types)
            else:
                subject_data.anat = \
                    normalize_result.outputs.normalized_files

        # collect node output
        subject_data.nipype_results['normalize_%s' % brain_name
                                    ] = normalize_result
        if brain_name == 'func':
            subject_data.func = unravel_filenames(
                normalize_result.outputs.normalized_files, file_types)
        else:
            subject_data.anat = normalize_result.outputs.normalized_files

    # commit output files
    if hardlink_output: subject_data.hardlink_output_files()

    # generate thumbnails
    if report:
        subject_data.generate_normalization_thumbnails()

    # explicit smoothing
    if np.sum(fwhm) + np.sum(anat_fwhm) > 0:
        subject_data = _do_subject_smooth(
            subject_data, fwhm, anat_fwhm=anat_fwhm, caching=caching,
            report=report
            )

    # commit output files
    if hardlink_output: subject_data.hardlink_output_files()

    return subject_data.sanitize()


def _do_subject_smooth(subject_data, fwhm, anat_fwhm=None, caching=True,
                       report=True, hardlink_output=True, software="spm"):
    """
    Wrapper for running spm.Smooth with optional reporting.

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functional images (subject_data.func) are to be
        smoothed

    caching: bool, optional (default True)
        if true, then caching will be enabled

    software: string, optional (default "spm")
        software to use for realignment; can be "spm", "fsl", or "python"

    Returns
    -------
    subject_data: `SubjectData` object
        preprocessed subject_data. The func field (subject_data.func) now
        contains the smoothed functional images of the subject

        New Attributes
        ==============
        subject_data.nipype_results['smooth']: Nipype output object
            (raw) result of running spm.Smooth

    Notes
    -----
    Input subject_data is modified.

    """

    # sanitize software choice
    software = software.lower()
    if software != "spm":
        raise NotImplementedError("Only SPM is supported; got '%s'" % software)

    # sanitize subject_data (do things like .nii.gz -> .nii conversion, etc.)
    subject_data.sanitize(niigz2nii=(software == "spm"))

    # prepare for smart caching
    if caching:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        smooth = NipypeMemory(base_dir=cache_dir).cache(spm.Smooth)
    else:
        smooth = spm.Smooth().run

    # run node
    # failed node ?
    subject_data.nipype_results['smooth'] = {}

    for brain_name, width in zip(
        ['func', 'anat'],
        [fwhm] + [anat_fwhm] * 7):
        brain = getattr(subject_data, brain_name)
        if not brain: continue
        print brain_name
        if not np.sum(width): continue
        in_files = brain
        if brain_name == "func":
            in_files, file_types = ravel_filenames(brain)
        if brain_name == "anat":
            anat_like = ['anat',
                         'mwgm', 'mwwm', 'mwcsf'  # normalized TPMs
                         ]
            anat_like = [x for x in anat_like if hasattr(subject_data, x)]
            in_files = [getattr(subject_data, x) for x in anat_like]

        smooth_result = smooth(in_files=in_files,
                               fwhm=width,
                               ignore_exception=False
                               )

        # failed node ?
        subject_data.nipype_results['smooth'][brain_name] = smooth_result
        if smooth_result.outputs is None:
            subject_data.failed = True
            warnings.warn("Failed smoothing %s" % brain_name)
            return subject_data

        brain = smooth_result.outputs.smoothed_files
        if brain_name == "func":
            brain = unravel_filenames(brain, file_types)
            subject_data.func = brain
        if brain_name == "anat":
            for j, x in enumerate(anat_like):
                setattr(subject_data, x, brain[j])

    # commit output files
    if hardlink_output: subject_data.hardlink_output_files()

    # reporting
    if report: subject_data.generate_smooth_thumbnails()

    return subject_data.sanitize()


def _do_subject_dartelnorm2mni(subject_data,
                               template_file,
                               fwhm=0,
                               anat_fwhm=0.,
                               output_modulated_tpms=False,
                               caching=True,
                               report=True,
                               parent_results_gallery=None,
                               cv_tc=True,
                               last_stage=True,
                               func_write_voxel_sizes=None,
                               anat_write_voxel_sizes=None,
                               hardlink_output=True
                               ):
    """
    Uses spm.DARTELNorm2MNI to warp subject brain into MNI space.

    Parameters
    ----------
    output_dir: string
        existing directory; results will be cache here

    output_modulated_tpms: bool, optional (default False)
        if set, then modulated TPMS will be produced (alongside unmodulated
        TPMs); this can be useful for VBM

    caching: bool, optional (default True)
        if true, then caching will be enabled

    **dartelnorm2mni_kargs: parameter-value list
        options to be passes to spm.DARTELNorm2MNI back-end

    """

    # sanitize subject_data (do things like .nii.gz -> .nii conversion, etc.)
    subject_data.sanitize(niigz2nii=True)

    # prepare for smart caching
    if caching:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        subject_data.mem = NipypeMemory(base_dir=cache_dir) 
        dartelnorm2mni = subject_data.mem.cache(
            spm.DARTELNorm2MNI)
        createwarped = subject_data.mem.cache(
            spm.CreateWarped)
    else:
        dartelnorm2mni = spm.DARTELNorm2MNI().run
        createwarped = spm.CreateWarped().run

    # warp subject tissue class images
    # into MNI space
    tricky_kwargs = {}
    if not anat_write_voxel_sizes is None:
        tricky_kwargs['voxel_size'] = compute_output_voxel_size(
           subject_data.anat, anat_write_voxel_sizes)
    for tissue in ['gm', 'wm']:
        if hasattr(subject_data, tissue):
            dartelnorm2mni_result = dartelnorm2mni(
                apply_to_files=getattr(subject_data, tissue),
                flowfield_files=subject_data.dartel_flow_fields,
                template_file=template_file,
                modulate=output_modulated_tpms,  # don't modulate
                fwhm=anat_fwhm,
                **tricky_kwargs
                )
            setattr(subject_data, "w" + tissue,
                    dartelnorm2mni_result.outputs.normalized_files)

    # warp anat into MNI space
    dartelnorm2mni_result = dartelnorm2mni(
        apply_to_files=subject_data.anat,
        flowfield_files=subject_data.dartel_flow_fields,
        template_file=template_file,
        ignore_exception=False,
        modulate=output_modulated_tpms,
        fwhm=anat_fwhm,
        **tricky_kwargs
        )
    subject_data.anat = dartelnorm2mni_result.outputs.normalized_files

    # warp functional image into MNI space
    # functional_file = do_3Dto4D_merge(functional_file)
    if subject_data.func:
        func_write_voxel_sizes = compute_output_voxel_size(
            subject_data.func, func_write_voxel_sizes)
        createwarped_result = createwarped(
            image_files=subject_data.func,
            flowfield_files=subject_data.dartel_flow_fields,
            ignore_exception=False
            )
        subject_data.func = createwarped_result.outputs.warped_files

        # resample func if necessary
        if not func_write_voxel_sizes is None:
            vox_dims = get_vox_dims(subject_data.func[0])
            if func_write_voxel_sizes != vox_dims:
                _resample_img = lambda input_filename: resample_img(
                    input_filename, func_write_voxel_sizes,
                    output_filename=os.path.join(
                        os.path.dirname(input_filename),
                        "resampled_" + os.path.basename(input_filename)))

                func = []
                for sess_func in subject_data.func:
                    assert get_vox_dims(sess_func) == vox_dims
                    func.append(_resample_img(sess_func) if isinstance(
                            sess_func, basestring) else [_resample_img(x)
                                                         for x in sess_func])
                subject_data.func = func

        # smooth func
        if np.sum(fwhm) > 0:
            subject_data = _do_subject_smooth(subject_data, fwhm,
                                              caching=caching,
                                              report=report
                                              )

    # hardlink output files
    if hardlink_output: subject_data.hardlink_output_files()

    if report:
        # generate normalization thumbnails
        subject_data.generate_normalization_thumbnails()

        # finalize
        subject_data.finalize_report(
            parent_results_gallery=parent_results_gallery,
            last_stage=last_stage)

    return subject_data.sanitize()


def do_subject_preproc(
    subject_data,
    deleteorient=False,

    slice_timing=False,
    slice_order="ascending",
    interleaved=False,
    refslice=1,
    TR=None,
    TA=None,
    slice_timing_software="spm",

    realign=True,
    realign_reslice=False,
    register_to_mean=True,
    realign_software="spm",

    coregister=True,
    coregister_reslice=False,
    coreg_anat_to_func=False,
    coregister_software="spm",

    segment=True,

    normalize=True,
    dartel=False,
    fwhm=0.,
    anat_fwhm=0.,
    func_write_voxel_sizes=None,
    anat_write_voxel_sizes=None,

    hardlink_output=True,
    report=True,
    cv_tc=True,
    parent_results_gallery=None,
    last_stage=True,
    preproc_undergone=None,
    prepreproc_undergone="",
    generate_preproc_undergone=True,
    caching=True,

    **kwargs
    ):
    """
    Function preprocessing data for a single subject.

    Parameters
    ----------
    subject_data: instance of `SubjectData`
        Object containing information about the subject under inspection
        (path to anat image, func image(s), output directory, etc.). Refer
        to documentation of `SubjectData` class

    realign: bool, optional (default True)
        if set, then the functional data will be realigned to correct for
        head-motion.

        subject_data.nipype_results['realign'] will contain the result from
        the spm.Realign node.

    coreg: bool, optional (default True)
        if set, the functional (subject_data.func) and anatomical
        (subject_data.anat) images will be corregistered. If this
        not set, and subject_data.anat is not None, the it is assumed that
        subject_data.func and subject_data.anat have already bean coregistered.

        subject_data.nipype_results['coregister'] will contain the result
        from the spm.Coregister node.

    coreg_anat_to_func: bool, optional (default False)
       if set, then functional data (subject_data.func) will be the reference
       during coregistration. By default the anatomical data
       (subject_data.anat) if the reference, to ensure a precise registration
       (since anatomical data has finer resolution)

    segment: bool, optional (default True)
        if set, then the subject's anatomical (subject_data.anat) image will be
        segmented to produce GM, WM, and CSF compartments (useful for both
        indirect normalization (intra-subject) or DARTEL (inter-subject) alike

        subject_data.nipype_results['segment'] will contain the result from
        the spm.Segment node.

    normalize: bool, optional (default True)
       if set, then the subject_data (subject_data.func and subject_data.anat)
       will will be warped into MNI space

       subject_data.nipype_results['normalize'] will contain the result from
       the spm.Normalize node.

    fwhm: float or list of 3 floats, optional (default 0)
        FWHM for smoothing the functional data (subject_data.func).
        If normalize is set, then this parameter is based to spm.Normalize,
        else spm.Smooth is used to explicitly smooth the functional data.

        If spm.Smooth is used for smoothing (i.e if normalize if False
        and fwhm is not 0), then subject_data.nipype_results['smooth']
        will contain the result from the spm.Smooth node.

    dartel: bool, optional (default False)
        flag indicating whether DARTEL will be chained with the results
        of this function

    deleteorient: bool (optional)
        if true, then orientation meta-data in all input image files for this
        subject will be stripped-off

    hardlink_output: bool, optional (default True)
        if set, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    cv_tc: bool (optional)
        if set, a summarizing the time-course of the coefficient of variation
        in the preprocessed fMRI time-series will be generated

    report: bool, optional (default True)
        if set, then HTML reports will be generated

    See also
    ========
    pypreprocess.purepython_preproc_utils

    """

    # disable nodes that can't run
    if not subject_data.func:
        slice_timing = realign = coregister = False
        fwhm = 0.
    if not subject_data.anat: anat_fwhm = 0

    assert not SPM_DIR is None and os.path.isdir(SPM_DIR), (
        "SPM_DIR '%s' doesn't exist; you need to export it!" % SPM_DIR)

    # sanitze subject data
    if isinstance(subject_data, dict):
        subject_data = SubjectData(**subject_data)
    else:
        assert isinstance(subject_data, SubjectData), (
            "subject_datta must be SubjectData instance or dict, "
            "got %s" % type(subject_data))

    subject_data.sanitize(deleteorient=deleteorient, niigz2nii=True)
    subject_data.failed = False

    # use EPI template for anat if anat is None
    if subject_data.anat is None:
        coregister = True
        segment = False
        subject_data.hires = EPI_TEMPLATE
        coreg_anat_to_func = False

    # sanitize fwhms
    fwhm = sanitize_fwhm(fwhm)
    anat_fwhm = sanitize_fwhm(anat_fwhm)

    # XXX For the moment, we can neither segment nor normalize without anat.
    # A trick would be to register the func with an EPI template and then
    # the EPI template to MNI
    segment = (not subject_data.anat is None) and segment

    # get ready for reporting
    if report:
        # generate explanation of preproc steps undergone by subject
        preproc_undergone = generate_preproc_undergone_docstring(
            dcm2nii=subject_data.isdicom,
            deleteorient=deleteorient,
            slice_timing=slice_timing,
            realign=realign,
            coregister=coregister,
            segment=segment,
            normalize=normalize,
            fwhm=fwhm, anat_fwhm=anat_fwhm,
            dartel=dartel,
            coreg_func_to_anat=not coreg_anat_to_func,
            prepreproc_undergone=prepreproc_undergone,
            has_func=subject_data.func
            )

        # initialize report factory
        subject_data.init_report(parent_results_gallery=parent_results_gallery,
                                 preproc_undergone=preproc_undergone,
                                 cv_tc=cv_tc)

    #############################
    # Slice-Timing Correction
    #############################
    if slice_timing:
        subject_data = _do_subject_slice_timing(
            subject_data, TR, refslice=refslice,
            TA=TA, slice_order=slice_order, interleaved=interleaved,
            report=report,  # post-stc reporting bugs like hell!
            software=slice_timing_software,
            hardlink_output=hardlink_output
            )

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    #######################
    #  motion correction
    #######################
    if realign:
        subject_data = _do_subject_realign(
            subject_data, caching=caching,
            reslice=realign_reslice,
            register_to_mean=register_to_mean,
            report=report,
            hardlink_output=hardlink_output,
            software=realign_software
            )

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    ##################################################################
    # co-registration of structural (anatomical) against functional
    ##################################################################
    if coregister:
        subject_data = _do_subject_coregister(
            subject_data, caching=caching,
            coreg_anat_to_func=coreg_anat_to_func,
            reslice=coregister_reslice,
            report=report,
            hardlink_output=hardlink_output,
            software=coregister_software
            )

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    #####################################
    # segmentation of anatomical image
    #####################################
    if segment:
        subject_data = _do_subject_segment(
            subject_data, caching=caching,
            normalize=normalize,
            report=report,
            hardlink_output=hardlink_output
            )

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    ##########################
    # Spatial Normalization
    ##########################
    if normalize:
        subject_data = _do_subject_normalize(
            subject_data,
            fwhm,  # smooth func after normalization
            anat_fwhm=anat_fwhm,
            func_write_voxel_sizes=func_write_voxel_sizes,
            anat_write_voxel_sizes=anat_write_voxel_sizes,
            caching=caching,
            report=report,
            hardlink_output=hardlink_output
            )

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    #########################################
    # Smooth without Spatial Normalization
    #########################################
    if not normalize and (np.sum(fwhm) + np.sum(anat_fwhm) > 0):
        subject_data = _do_subject_smooth(subject_data,
                                          fwhm, anat_fwhm=anat_fwhm,
                                          caching=caching,
                                          report=report,
                                          hardlink_output=hardlink_output
                                          )

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    # hard-link node output files
    if last_stage or not dartel:
        if hardlink_output: subject_data.hardlink_output_files(final=True)

    if report and not dartel:
        subject_data.finalize_report(last_stage=last_stage)

    # return preprocessed subject_data
    return subject_data.sanitize()


def _do_subjects_dartel(subjects,
                        output_dir,
                        fwhm=0,
                        anat_fwhm=0.,
                        func_write_voxel_sizes=None,
                        anat_write_voxel_sizes=None,
                        output_modulated_tpms=False,
                        n_jobs=-1,
                        report=True,
                        cv_tc=True,
                        parent_results_gallery=None,
                        **kwargs
                        ):
    """
    Runs NewSegment + DARTEL on given subjects.

    """

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = NipypeMemory(base_dir=cache_dir)

    # compute gm, wm, etc. structural segmentation using Newsegment
    # XXX verify that the following TPM paths remain valid in case of
    # precompilted SPM
    newsegment = mem.cache(spm.NewSegment)
    tissue1 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 1),
               2, (True, True), (False, False))
    tissue2 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 2),
               2, (True, True), (False, False))
    tissue3 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 3),
               2, (True, False), (False, False))
    tissue4 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 4),
               3, (False, False), (False, False))
    tissue5 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 5),
               4, (False, False), (False, False))
    tissue6 = ((os.path.join(SPM_DIR, 'toolbox/Seg/TPM.nii'), 6),
               2, (False, False), (False, False))
    newsegment_result = newsegment(
        channel_files=[subject_data.anat for subject_data in subjects],
        tissues=[tissue1, tissue2, tissue3, tissue4, tissue5, tissue6],
        ignore_exception=False
        )
    if newsegment_result.outputs is None: return

    # compute DARTEL template for group data
    dartel = mem.cache(spm.DARTEL)
    dartel_input_images = [tpms for tpms in
                           newsegment_result.outputs.dartel_input_images
                           if tpms]
    dartel_result = dartel(
        image_files=dartel_input_images,)
    if dartel_result.outputs is None: return

    for subject_data, j in zip(subjects, xrange(len(subjects))):
        subject_data.gm = newsegment_result.outputs.dartel_input_images[0][j]
        subject_data.wm = newsegment_result.outputs.dartel_input_images[1][j]
        subject_data.dartel_flow_fields = dartel_result.outputs\
            .dartel_flow_fields[j]

    # warp individual brains into group (DARTEL) space
    preproc_subject_data = Parallel(
        n_jobs=n_jobs, verbose=100,
        pre_dispatch='1.5 * n_jobs',
        )(delayed(
            _do_subject_dartelnorm2mni)(
                subject_data,
                report=report,
                cv_tc=cv_tc,
                parent_results_gallery=parent_results_gallery,
                fwhm=fwhm, anat_fwhm=anat_fwhm,
                func_write_voxel_sizes=func_write_voxel_sizes,
                anat_write_voxel_sizes=anat_write_voxel_sizes,
                output_modulated_tpms=output_modulated_tpms,
                template_file=dartel_result.outputs.final_template_file,
                )
          for subject_data in subjects)

    return preproc_subject_data, newsegment_result


def do_subjects_preproc(subject_factory,
                        **preproc_params
                        ):
    """
    This function does intra-subject preprocessing on a group of subjects.

    Parameters
    ----------
    subject_factory: iterable of `SubjectData` objects
        data for the subjects to be preprocessed

    output_dir: string, optional (default None)
        output directory where all results will be written

    n_jobs: int, optional (default None)
        number of jobs to create; parameter passed to `joblib.Parallel`.
        if N_JOBS is defined in the shell environment, then its value
        is used instead.

    caching: bool, optional (default True)
       if set then caching (joblib, nipype, etc.), will by used where ever
       useful

    dartel: bool, optional (default False)
        flag indicating whether NewSegment + DARTEL should used for
        normalization

    func_write_voxel_sizes: triplet of floats, optional (default None)
        final voxel size of all functional images after normalization

    anat_wrirt_voxel_sizes: triple of floats, optional (default None)
        final voxel size for all anatomical images after normalization

    report: bool, optional (default True)
        if set, then HTML reports will be generated

    dataset_id: string, optional (default None)
        brief description of the dataset being preprocessed
        (e.g "ABIDE", "NYU")

    dataset_description: string, optional (default None)
        longer description of what the dataset being preprocessed is
        all about

    prepreproc_undergone: string, optional (default None)
        preprocessed already undergone by the dataset out-side this function

    hardlink_output: bool, optional (default True)
        indicates whether inter-mediate output files should be linked subjects'
        output (immediate) directories (by default, only final output files
        are linked)

    preproc_params: parameter-value dict
        optional parameters passed to the \do_subject_preproc` API. See
        the API documentation for details

    matlab_exec: string, optional (default None)
        full path to the MATLAB executable to be used with SPM

    spm_dir: string, optional (default None)
        full path to the SPM installation directory to be used

    Returns
    -------
    preproc_subject_data: list of preprocessed `SubjectData` objects
        each elements is the preprocessed version of the corresponding
        input `SubjectData` object

    """

    # load .ini ?
    preproc_details = None
    if isinstance(subject_factory, basestring):
        with open(subject_factory, "r") as fd:
            preproc_details = fd.read()
            fd.close()
        subject_factory, _preproc_params = _generate_preproc_pipeline(
            subject_factory, dataset_dir=preproc_params.get(
                "dataset_dir", None))
        _preproc_params.update(preproc_params)
        preproc_params = _preproc_params

    # collect some params for local usage
    dartel = preproc_params.get('dartel', False)
    output_dir = preproc_params.get('output_dir', "pypreprocess_output")
    if "output_dir" in preproc_params: del preproc_params["output_dir"]
    report = preproc_params.get("report", True)
    spm_dir = preproc_params.get("spm_dir", None)
    matlab_exec = preproc_params.get("matlab_exec", None)
    dataset_id = preproc_params.get('dataset_id', None)
    dataset_description = preproc_params.get('dataset_description', None)
    shutdown_reloaders = preproc_params.get('shutdown_reloaders', True)
    n_jobs = preproc_params.get('n_jobs', None)
    if "n_jobs" in preproc_params: del preproc_params["n_jobs"]

    print "Using the following parameters for preprocessing:"
    for k, v in preproc_params.iteritems(): print "\t%s=%s" % (k, v)

    # generate subjects (if generator)
    subject_factory = [subject_data for subject_data in subject_factory]

    # DARTEL on 1 subject is senseless
    dartel = dartel and (len(subject_factory) > 1)

    # configure SPM back-end
    # XXX what about precompiled SPM; the following check would be too harsh in
    # XXX this case
    _configure_backends(spm_dir=spm_dir, matlab_exec=matlab_exec)
    assert not SPM_DIR is None and os.path.isdir(SPM_DIR), (
        "SPM_DIR '%s' doesn't exist; you need to export it!" % SPM_DIR)

    # configure number of jobs
    if n_jobs is None: n_jobs = 1
    n_jobs = int(os.environ.get('N_JOBS', n_jobs))

    # sanitize output_dir
    if output_dir is None:
        if "output_dir" in preproc_params:
            output_dir = preproc_params["output_dir"]
            del preproc_params['output_dir']
        else: output_dir = os.path.join(os.getcwd(), "pypreprocess_results")

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # generate list of subjects
    subjects = [subject_data for subject_data in subject_factory]

    # ensure that we actually have data to preprocess
    assert len(subjects) > 0, "subject_factory is empty; nothing to do!"

    # sanitize subject output directories
    for subject_data in subjects:
        if not hasattr(subject_data, "output_dir"):
            if not hasattr(subject_data.subject_id):
                subject_data.subject_id = "sub001"
            subject_data.output_dir = os.path.join(output_dir,
                                                   subject_data.subject_id)

    # generate html report (for QA) as desired
    parent_results_gallery = None
    if report:
        # what exactly was typed at the command-line (terminal) ?
        command_line = "python %s" % " ".join(sys.argv)

        # copy web stuff to output_dir
        copy_web_conf_files(output_dir)

        report_log_filename = os.path.join(output_dir, 'report_log.html')
        report_preproc_filename = os.path.join(
            output_dir, 'report_preproc.html')
        report_filename = os.path.join(output_dir, 'report.html')

        # get caller module handle from stack-frame
        user_script_name = sys.argv[0]
        user_source_code = get_module_source_code(
            user_script_name)

        # scrape complete configuration used for preprocessing
        if preproc_details is None:
            preproc_details = ""
            frame = inspect.currentframe()
            args, _, _, values = inspect.getargvalues(frame)
            preproc_func_name = inspect.getframeinfo(frame)[2]
            preproc_details += ("Function <i>%s(...)</i> was invoked by "
                                "the script"
                                " <i>%s</i> with the following arguments:"
                                ) % (preproc_func_name, user_script_name)
            args_dict = dict((arg, values[arg]) for arg in args if not arg in [
                    "dataset_description",
                    "report_filename",
                    "report",
                    "cv_tc",
                    "shutdown_reloaders",
                    "subjects",
                    # add other args to exclude below
                    ])
            args_dict['output_dir'] = output_dir
            preproc_details += dict_to_html_ul(args_dict)
        details_filename = os.path.join(output_dir, "preproc_details.html")
        open(details_filename, "w").write("<pre>%s</pre>" % preproc_details)

        # initialize results gallery
        loader_filename = os.path.join(output_dir, "results_loader.php")
        parent_results_gallery = ResultsGallery(
            loader_filename=loader_filename, refresh_timeout=30)

        # initialize progress bar
        progress_logger = ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename, report_preproc_filename])

        # html markup
        log = get_dataset_report_log_html_template(
            start_time=time.ctime())

        preproc = get_dataset_report_preproc_html_template(
            results=parent_results_gallery,
            start_time=time.ctime(),
            # preproc_undergone=preproc_undergone,
            dataset_description=dataset_description,
            source_code=user_source_code,
            source_script_name=user_script_name,
            )

        main_html = get_dataset_report_html_template(
            results=parent_results_gallery,
            start_time=time.ctime(),
            dataset_id=dataset_id
            )

        with open(report_log_filename, 'w') as fd:
            fd.write(str(log))
            fd.close()
        with open(report_preproc_filename, 'w') as fd:
            fd.write(str(preproc))
            fd.close()
        with open(report_filename, 'w') as fd:
            fd.write(str(main_html))
            fd.close()

        if not dartel: preproc_params['parent_results_gallery'
                                      ] = parent_results_gallery

        # log command line
        progress_logger.log("<b>Command line</b><br/>")
        progress_logger.log("%s" % command_line)
        progress_logger.log("<hr/>")

        # log environ variables
        progress_logger.log("<b>Environ Variables</b><br/>")
        progress_logger.log(
            "<ul>" + "".join(["<li>%s: %s</li>" % (item, value)
                              for item, value in os.environ.iteritems()])
            + "</ul><hr/>")

    def finalize_report():
        if not report: return

        progress_logger.finish(report_preproc_filename)

        if shutdown_reloaders:
            print "Finishing %s..." % output_dir
            progress_logger.finish_dir(output_dir)

        print "\r\n\tHTML report written to %s" % report_preproc_filename

    # don't yet segment nor normalize if dartel enabled
    if dartel:
        for item in ["segment", "normalize", "cv_tc", "last_stage"]:
            preproc_params[item] = False

    # run classic preproc
    preproc_subject_data = Parallel(n_jobs=n_jobs)(delayed(do_subject_preproc)(
            subject_data, **preproc_params) for subject_data in subjects)

    # run DARTEL
    if dartel:
        preproc_subject_data = _do_subjects_dartel(
            preproc_subject_data, output_dir,
            n_jobs=n_jobs,
            parent_results_gallery=parent_results_gallery,
            **preproc_params)

    finalize_report()

    return preproc_subject_data
