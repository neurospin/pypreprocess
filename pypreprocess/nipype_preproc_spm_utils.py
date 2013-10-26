"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

XXX TODO : if a node fails and reporting is enabled, the generate a
failure thumbnail!!!

"""

# standard imports
import numpy as np
import nibabel
import os
import sys
import time
from matplotlib.pyplot import cm
import inspect

# import joblib API
from joblib import (Parallel,
                    delayed,
                    Memory as JoblibMemory
                    )

# import nipype API
import nipype.interfaces.spm as spm
from nipype.caching import Memory as NipypeMemory

# import API for i/o
from .io_utils import (load_specific_vol,
                       ravel_filenames,
                       unravel_filenames,
                       get_vox_dims,
                       niigz2nii
                       )
from subject_data import SubjectData

# import API for reporting
from .reporting.base_reporter import (
    ResultsGallery,
    ProgressReport,
    copy_web_conf_files,
    get_module_source_code,
    dict_to_html_ul,
    )
from .reporting.preproc_reporter import (
    generate_preproc_undergone_docstring,
    get_dataset_report_log_html_template,
    get_dataset_report_preproc_html_template,
    get_dataset_report_html_template
    )

# configure SPM
from .configure_spm import SPM_DIR

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


def _do_subject_realign(subject_data, nipype_mem=None,
                        do_report=True):
    """
    Wrapper for running spm.Realign with optional reporting.

    If subject_data has a `results_gallery` attribute, then QA thumbnails will
    be commited after this node is executed

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functional images (subject_data.func) are to be
        realigned.

    nipype_mem: `nipype.caching.Memory` object
        Wrapper for running node with nipype.caching.Memory

    do_report: bool, optional (default True)
       flag controlling whether post-preprocessing reports should be generated

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

    if not hasattr(subject_data, 'nipype_results'):
        subject_data.nipype_results = {}

    # .nii.gz -> .nii
    subject_data.niigz2nii()

    # prepare for smart caching
    if nipype_mem is None:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        nipype_mem = NipypeMemory(base_dir=cache_dir)

    # configure node
    realign = nipype_mem.cache(spm.Realign)

    # run node
    realign_result = realign(
        in_files=subject_data.func,
        jobtype="estwrite",
        )

    subject_data.nipype_results['realign'] = realign_result

    # failed node
    if realign_result.outputs is None:
        subject_data.failed = True
        return subject_data

    # collect output
    subject_data.func = realign_result.outputs.realigned_files
    subject_data.realignment_parameters = \
        realign_result.outputs.realignment_parameters

    # generate realignment thumbs
    if do_report:
        subject_data.generate_realignment_thumbnails()

    return subject_data


def _do_subject_coregister(subject_data, jobtype="estimate",
                           coreg_anat_to_func=False,
                           nipype_mem=None, joblib_mem=None,
                           do_report=True
                           ):
    """
    Wrapper for running spm.Coregister with optional reporting.

    If subject_data has a `results_gallery` attribute, then QA thumbnails will
    be commited after this node is executed

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functional and anatomical images (subject_data.func
        and subject_data.anat) are to be coregistered

    nipype_cached: nipype memroy cache
        Wrapper for running node with nipype.caching.Memory

    coreg_anat_to_func: bool, optional (default False)
       if set, then functional data (subject_data.func) will be the reference
       during coregistration. By default the anatomical data
       (subject_data.anat) if the reference, to ensure a precise registration
       (since anatomical data has finer resolution)

    do_report: bool, optional (default True)
       flag controlling whether post-preprocessing reports should be generated

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

    # .nii.gz -> .nii
    subject_data.niigz2nii()

    # prepare for smart caching
    if nipype_mem is None:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        nipype_mem = NipypeMemory(base_dir=cache_dir)

    if joblib_mem is None:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

            joblib_mem = JoblibMemory(base_dir=cache_dir)

    def _save_vol(data, affine, output_filename):
        nibabel.save(nibabel.Nifti1Image(data, affine), output_filename)

    # config node
    coreg = nipype_mem.cache(spm.Coregister)
    apply_to_files = []
    ref_brain = 'anat'
    src_brain = 'func'

    if coreg_anat_to_func:
        ref_brain, src_brain = src_brain, ref_brain
        coreg_target = subject_data.func[0]
        if subject_data.anat is None:
            if not subject_data.hires is None:
                coreg_source = subject_data.hires
        else:
            coreg_source = subject_data.anat
    else:
        coreg_target = subject_data.anat
        ref_func = joblib_mem.cache(load_specific_vol)(
            subject_data.func if isinstance(subject_data.func, basestring)
            else subject_data.func[0], 0)[0]
        coreg_source = os.path.join(subject_data.tmp_output_dir,
                                "_coreg_first_func_vol.nii")
        joblib_mem.cache(_save_vol)(ref_func.get_data(),
                                    ref_func.get_affine(),
                                    coreg_source)

        apply_to_files, file_types = ravel_filenames(subject_data.func)

    # run node
    coreg_result = coreg(target=coreg_target,
                         source=coreg_source,
                         apply_to_files=apply_to_files,
                         jobtype=jobtype,
                         ignore_exception=True
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
        subject_data.func = unravel_filenames(
            coreg_result.outputs.coregistered_files, file_types)

    # generate coregistration thumbs
    if do_report:
        subject_data.generate_coregistration_thumbnails()

    return subject_data


def _do_subject_segment(subject_data, do_normalize=False, nipype_mem=None,
                        do_report=True):
    """
    Wrapper for running spm.Segment with optional reporting.

    If subject_data has a `results_gallery` attribute, then QA thumbnails will
    be commited after this node is executed

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose anatomical image (subject_data.anat) is to be
        segmented

    nipype_mem: `nipype.caching.Memory` object
        Wrapper for running node with nipype.caching.Memory

    do_normalize: bool, optional (default False)
        flag indicating whether warped brain compartments (gm, wm, csf) are to
        be generated (necessary if the caller wishes the brain later)

    do_report: bool, optional (default True)
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

        if do_normalize then the following additional data fiels are
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

    # .nii.gz -> .nii
    subject_data.niigz2nii()

    # prepare for smart caching
    if nipype_mem is None:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        nipype_mem = NipypeMemory(base_dir=cache_dir)

    # configure node
    segment = nipype_mem.cache(spm.Segment)
    if not do_normalize:
        gm_output_type = [False, False, True]
        wm_output_type = [False, False, True]
        csf_output_type = [False, False, True]
    else:
        gm_output_type = [True, True, True]
        wm_output_type = [True, True, True]
        csf_output_type = [True, True, True]

    # run node
    segment_result = segment(
        data=subject_data.anat,
        gm_output_type=wm_output_type,
        wm_output_type=gm_output_type,
        csf_output_type=csf_output_type,
        tissue_prob_maps=[GM_TEMPLATE, WM_TEMPLATE, CSF_TEMPLATE],
        gaussians_per_class=[2, 2, 2, 4],
        affine_regularization="none",
        bias_regularization=0.0001,
        bias_fwhm=60,
        warping_regularization=1,
        ignore_exception=True
        )

    # failed node
    subject_data.nipype_results['segment'] = segment_result
    if segment_result.outputs is None:
        subject_data.failed = True
        return subject_data

    # collect output
    subject_data.nipype_results['segment'] = segment_result
    subject_data.gm = segment_result.outputs.native_gm_image
    subject_data.wm = segment_result.outputs.native_wm_image
    subject_data.csf = segment_result.outputs.native_csf_image
    if do_normalize:
        subject_data.wgm = segment_result.outputs.normalized_gm_image
        subject_data.wwm = segment_result.outputs.normalized_wm_image
        subject_data.wcsf = segment_result.outputs.normalized_csf_image

    # generate segmentation thumbs
    if do_report:
        subject_data.generate_segmentation_thumbnails()

    return subject_data


def _do_subject_normalize(subject_data, fwhm=0., nipype_mem=None,
                          func_write_voxel_sizes=None,
                          anat_write_voxel_sizes=None,
                          do_report=True
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

    nipype_cached: nipype memroy cache
        Wrapper for running node with nipype.caching.Memory

    fwhm: float or list of 3 floats, optional (default 0)
        FWHM for smoothing the functional data (subject_data.func).
        If do_normalize is set, then this parameter is based to spm.Normalize,
        else spm.Smooth is used to explicitly smooth the functional data.

        If spm.Smooth is used for smoothing (i.e if do_normalize if False
        and fwhm is not 0), then subject_data.nipype_results['smooth']
        will contain the result from the spm.Smooth node.

    do_report: bool, optional (default True)
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

    # .nii.gz -> .nii
    subject_data.niigz2nii()

    # prepare for smart caching
    if nipype_mem is None:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        nipype_mem = NipypeMemory(base_dir=cache_dir)

    segmented = 'segment' in subject_data.nipype_results

    # configure node for normalization
    if not segmented:
        # learn T1 deformation without segmentation
        t1_template = niigz2nii(SPM_T1_TEMPLATE,
                                output_dir=subject_data.output_dir)
        normalize = nipype_mem.cache(spm.Normalize)
        normalize_result = normalize(source=subject_data.anat,
                                     template=t1_template,
                                     )
        parameter_file = normalize_result.outputs.normalization_parameters
        normalize = nipype_mem.cache(spm.Normalize)
    else:
        normalize = nipype_mem.cache(spm.Normalize)
        parameter_file = subject_data.nipype_results[
            'segment'].outputs.transformation_mat

    # do normalization proper
    for brain_name, brain, cmap in zip(
        ['anat', 'func'], [subject_data.anat, subject_data.func],
        [cm.gray, cm.spectral]):
        if segmented:
            if brain_name == 'func':
                apply_to_files, file_types = ravel_filenames(subject_data.func)
                if func_write_voxel_sizes is None:
                    write_voxel_sizes = get_vox_dims(apply_to_files)
                else:
                    write_voxel_sizes = func_write_voxel_sizes
            else:
                apply_to_files = subject_data.anat
                if anat_write_voxel_sizes is None:
                    write_voxel_sizes = get_vox_dims(apply_to_files)
                else:
                    write_voxel_sizes = anat_write_voxel_sizes
                apply_to_files = subject_data.anat

            # run node
            normalize_result = normalize(
                parameter_file=parameter_file,
                apply_to_files=apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=write_voxel_sizes,
                write_interp=1,
                jobtype='write',
                ignore_exception=True
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
                else:
                    write_voxel_sizes = func_write_voxel_sizes
            else:
                apply_to_files = subject_data.anat
                apply_to_files, file_types = ravel_filenames(subject_data.func)
                if anat_write_voxel_sizes is None:
                    write_voxel_sizes = get_vox_dims(apply_to_files)
                else:
                    write_voxel_sizes = anat_write_voxel_sizes

            # run node
            normalize_result = normalize(
                parameter_file=parameter_file,
                apply_to_files=apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=write_voxel_sizes,
                write_wrap=[0, 0, 0],
                write_interp=1,
                jobtype='write',
                ignore_exception=True
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

    # generate thumbnails
    if do_report:
        subject_data.generate_normalization_thumbnails()

    # explicit smoothing
    if np.sum(fwhm) > 0:
        subject_data = _do_subject_smooth(
            subject_data, fwhm, nipype_mem=nipype_mem,
            do_report=do_report
            )

    return subject_data


def _do_subject_smooth(subject_data, fwhm, nipype_mem=None, do_report=True):
    """
    Wrapper for running spm.Smooth with optional reporting.

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functional images (subject_data.func) are to be
        smoothed

    nipype_mem: `nipype.caching.Memory` object
        Wrapper for running node with nipype.caching.Memory

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

    # .nii.gz -> .nii
    subject_data.niigz2nii()

    # prepare for smart caching
    if nipype_mem is None:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        nipype_mem = NipypeMemory(base_dir=cache_dir)

    # configure node
    smooth = nipype_mem.cache(spm.Smooth)
    in_files, file_types = ravel_filenames(subject_data.func)

    # run node
    smooth_result = smooth(in_files=in_files,
                           fwhm=fwhm,
                           ignore_exception=False
                           )
    # failed node ?
    subject_data.nipype_results['smooth'] = smooth_result
    if smooth_result.outputs is None:
        subject_data.failed = True
        return subject_data

    # collect results
    subject_data.func = unravel_filenames(
        smooth_result.outputs.smoothed_files, file_types)

    # reporting
    if do_report:
        subject_data.generate_smooth_thumbnails()

    return subject_data


def _do_subject_dartelnorm2mni(subject_data,
                               template_file,
                               fwhm=0,
                               nipype_mem=None,
                               do_report=True,
                               parent_results_gallery=None,
                               do_cv_tc=True,
                               last_stage=True
                               ):
    """
    Uses spm.DARTELNorm2MNI to warp subject brain into MNI space.

    Parameters
    ----------
    output_dir: string
        existing directory; results will be cache here

    **dartelnorm2mni_kargs: parameter-value list
        options to be passes to spm.DARTELNorm2MNI back-end

    """

    # .nii.gz -> .nii
    subject_data.niigz2nii()

    # prepare for smart caching
    if nipype_mem is None:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        nipype_mem = NipypeMemory(base_dir=cache_dir)

    # configure node
    dartelnorm2mni = nipype_mem.cache(spm.DARTELNorm2MNI)

    # warp subject tissue class image (produced by Segment or NewSegment)
    # into MNI space
    for tissue in ['gm', 'wm']:
        if hasattr(subject_data, tissue):
            dartelnorm2mni_result = dartelnorm2mni(
                apply_to_files=getattr(subject_data, tissue),
                flowfield_files=subject_data.dartel_flow_fields,
                template_file=template_file,
                modulate=False  # don't modulate
                )
            setattr(subject_data, "w" + tissue,
                    dartelnorm2mni_result.outputs.normalized_files)

    # warp functional image into MNI space
    # functional_file = do_3Dto4D_merge(functional_file)
    createwarped = nipype_mem.cache(spm.CreateWarped)
    createwarped_result = createwarped(
        image_files=subject_data.func,
        flowfield_files=subject_data.dartel_flow_fields,
        ignore_exception=False
        )
    subject_data.func = createwarped_result.outputs.warped_files

    # warp anat into MNI space
    dartelnorm2mni_result = dartelnorm2mni(
        apply_to_files=subject_data.anat,
        flowfield_files=subject_data.dartel_flow_fields,
        template_file=template_file,
        ignore_exception=True,
        modulate=False,  # don't modulate
        fwhm=0.  # don't smooth
        )
    subject_data.anat = dartelnorm2mni_result.outputs.normalized_files

    # hardlink output files
    subject_data.hardlink_output_files()

    if do_report:
        # generate normalization thumbnails
        subject_data.generate_normalization_thumbnails()

        # finalize
        subject_data.finalize_report(
            parent_results_gallery=parent_results_gallery,
            last_stage=last_stage)

    return subject_data


def do_subject_preproc(
    subject_data,
    do_realign=True,
    do_coreg=True,
    coreg_anat_to_func=False,
    do_segment=True,
    do_normalize=True,
    do_dartel=False,
    do_cv_tc=True,
    fwhm=0.,
    func_write_voxel_sizes=None,
    anat_write_voxel_sizes=None,
    do_deleteorient=False,
    hardlink_output=True,
    do_report=True,
    parent_results_gallery=None,
    last_stage=True,
    preproc_undergone=None,
    prepreproc_undergone="",
    ):
    """
    Function preprocessing data for a single subject.

    Parameters
    ----------
    subject_data: instance of `SubjectData`
        Object containing information about the subject under inspection
        (path to anat image, func image(s), output directory, etc.). Refer
        to documentation of `SubjectData` class

    do_realign: bool, optional (default True)
        if set, then the functional data will be realigned to correct for
        head-motion.

        subject_data.nipype_results['realign'] will contain the result from
        the spm.Realign node.

    do_coreg: bool, optional (default True)
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

    do_segment: bool, optional (default True)
        if set, then the subject's anatomical (subject_data.anat) image will be
        segmented to produce GM, WM, and CSF compartments (useful for both
        indirect normalization (intra-subject) or DARTEL (inter-subject) alike

        subject_data.nipype_results['segment'] will contain the result from
        the spm.Segment node.

    do_normalize: bool, optional (default True)
       if set, then the subject_data (subject_data.func and subject_data.anat)
       will will be warped into MNI space

       subject_data.nipype_results['normalize'] will contain the result from
       the spm.Normalize node.

    fwhm: float or list of 3 floats, optional (default 0)
        FWHM for smoothing the functional data (subject_data.func).
        If do_normalize is set, then this parameter is based to spm.Normalize,
        else spm.Smooth is used to explicitly smooth the functional data.

        If spm.Smooth is used for smoothing (i.e if do_normalize if False
        and fwhm is not 0), then subject_data.nipype_results['smooth']
        will contain the result from the spm.Smooth node.

    do_dartel: bool, optional (default False)
        flag indicating whether DARTEL will be chained with the results
        of this function

    do_deleteorient: bool (optional)
        if true, then orientation meta-data in all input image files for this
        subject will be stripped-off

    hardlink_output: bool, optional (default True)
        if set, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    do_cv_tc: bool (optional)
        if set, a summarizing the time-course of the coefficient of variation
        in the preprocessed fMRI time-series will be generated

    do_report: bool, optional (default True)
        if set, then HTML reports will be generated

    See also
    ========
    pypreprocess.purepython_preproc_utils

    """

    # sanitze subject data
    if isinstance(subject_data, dict):
        subject_data = SubjectData(**subject_data)
    else:
        assert isinstance(subject_data, SubjectData), (
            "subject_datta must be SubjectData instance or dict, "
            "got %s" % type(subject_data))

    subject_data.sanitize(do_deleteorient=do_deleteorient, do_niigz2nii=True)
    subject_data.failed = False

    # can't coreg without anat
    do_coreg = do_coreg and not subject_data.anat is None
 
    # sanitize fwhm
    if not fwhm is None:
        if not np.shape(fwhm):
            fwhm = [fwhm, fwhm, fwhm]
        if len(fwhm) == 1:
            fwhm = list(fwhm) * 3
        else:
            assert len(fwhm) == 3, ("fwhm must be float or list of 3 "
                                    "floats; got %s" % fwhm)

    # XXX For the moment, we can neither segment nor normalize without anat.
    # A trick would be to register the func with an EPI template and then
    # the EPI template to MNI
    do_segment = (not subject_data.anat is None) and do_segment
    do_normalize = (not subject_data.anat is None) and do_normalize

    # nipype outputs
    subject_data.nipype_results = {}

    # prepare for smart caching
    cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    nipype_mem = NipypeMemory(base_dir=cache_dir)
    joblib_mem = JoblibMemory(cache_dir, verbose=100)

    # get ready for reporting
    if do_report:
        # generate explanation of preproc steps undergone by subject
        if preproc_undergone is None:
            preproc_undergone = generate_preproc_undergone_docstring(
                do_realign=do_realign,
                do_coreg=do_coreg,
                do_segment=do_segment,
                do_normalize=do_normalize,
                fwhm=fwhm,
                do_dartel=do_dartel,
                coreg_func_to_anat=not coreg_anat_to_func,
                prepreproc_undergone=prepreproc_undergone
                )

        # initialize reports factory
        subject_data.init_report(parent_results_gallery=parent_results_gallery,
                                 preproc_undergone=preproc_undergone,
                                 do_cv_tc=do_cv_tc)

    #######################
    #  motion correction
    #######################
    if do_realign:
        subject_data = _do_subject_realign(
            subject_data, nipype_mem=nipype_mem,
            do_report=do_report
            )

        # hard-link node output files
        if hardlink_output:
            subject_data.hardlink_output_files()

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    ##################################################################
    # co-registration of structural (anatomical) against functional
    ##################################################################
    if do_coreg:
        subject_data = _do_subject_coregister(
            subject_data, nipype_mem=nipype_mem,
            joblib_mem=joblib_mem,
            coreg_anat_to_func=coreg_anat_to_func,
            do_report=do_report
            )

        # hard-link node output files
        if hardlink_output:
            subject_data.hardlink_output_files()

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    #####################################
    # segmentation of anatomical image
    #####################################
    if do_segment:
        subject_data = _do_subject_segment(
            subject_data, nipype_mem=nipype_mem,
            do_normalize=do_normalize,
            do_report=do_report
            )

        # hard-link node output files
        if hardlink_output:
            subject_data.hardlink_output_files()

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    ##########################
    # Spatial Normalization
    ##########################
    if do_normalize:
        subject_data = _do_subject_normalize(
            subject_data,
            fwhm,  # smooth func after normalization
            func_write_voxel_sizes=func_write_voxel_sizes,
            anat_write_voxel_sizes=anat_write_voxel_sizes,
            nipype_mem=nipype_mem,
            do_report=do_report
            )

        # hard-link node output files
        if hardlink_output:
            subject_data.hardlink_output_files()

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    #########################################
    # Smooth without Spatial Normalization
    #########################################
    if not do_normalize and np.sum(fwhm) > 0:
        subject_data = _do_subject_smooth(subject_data, fwhm,
                                          nipype_mem=nipype_mem,
                                          do_report=do_report
                                          )

        # handle failed node
        if subject_data.failed:
            subject_data.finalize_report(last_stage=last_stage)
            return subject_data

    if do_report and not do_dartel:
        subject_data.finalize_report(last_stage=last_stage)

    # hard-link node output files
    if last_stage or not do_dartel:
        if hardlink_output:
            subject_data.hardlink_output_files(final=True)

    # return preprocessed subject_data
    return subject_data


def _do_subjects_dartel(subjects,
                        output_dir,
                        fwhm=0,
                        n_jobs=-1,
                        do_report=True,
                        do_cv_tc=True,
                        parent_results_gallery=None
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
        ignore_exception=True
        )

    if newsegment_result.outputs is None:
        return
    # compute DARTEL template for group data
    dartel = mem.cache(spm.DARTEL)
    dartel_input_images = [tpms for tpms in
                           newsegment_result.outputs.dartel_input_images
                           if tpms]
    dartel_result = dartel(
        image_files=dartel_input_images,)

    if dartel_result.outputs is None:
        return

    for subject_data, j in zip(subjects, xrange(len(subjects))):
        subject_data.gm = newsegment_result.outputs.dartel_input_images[0][j]
        subject_data.wm = newsegment_result.outputs.dartel_input_images[1][j]
        subject_data.dartel_flow_fields = dartel_result.outputs\
            .dartel_flow_fields[j]

    # warp individual brains into group (DARTEL) space
    preproc_subject_data = Parallel(
        n_jobs=n_jobs, verbose=100,
        pre_dispatch='1.5*n_jobs',  # for scalability over RAM
        )(delayed(
            _do_subject_dartelnorm2mni)(
                subject_data,
                do_report=do_report,
                do_cv_tc=do_cv_tc,
                parent_results_gallery=parent_results_gallery,
                fwhm=fwhm,
                template_file=dartel_result.outputs.final_template_file,
                )
          for subject_data in subjects)

    return preproc_subject_data, newsegment_result


def do_subjects_preproc(subject_factory,
                        output_dir=None,
                        hardlink_output=True,
                        n_jobs=None,
                        do_dartel=False,
                        do_report=True,
                        dataset_id="UNKNOWN!",
                        dataset_description="",
                        prepreproc_undergone="",
                        shutdown_reloaders=True,
                        **do_subject_preproc_kwargs
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

    do_dartel: bool, optional (default False)
        flag indicating whether NewSegment + DARTEL should used for
        normalization

    do_report: bool, optional (default True)
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

    do_subject_preproc_kwargs: parameter-value dict
        optional parameters passed to the \do_subject_preproc` API. See
        the API documentation for details

    Returns
    -------
    preproc_subject_data: list of preprocessed `SubjectData` objects
        each elements is the preprocessed version of the corresponding
        input `SubjectData` object

    """

    do_subject_preproc_kwargs['do_report'] = do_report
    do_subject_preproc_kwargs["do_dartel"] = do_dartel
    do_subject_preproc_kwargs['prepreproc_undergone'] = prepreproc_undergone
    do_subject_preproc_kwargs['hardlink_output'] = hardlink_output

    # sanitize output_dir
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "pypreproc_results")

    if not output_dir is None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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

    # configure number of jobs
    n_jobs = int(os.environ['N_JOBS']) if 'N_JOBS' in os.environ else (
        -1 if n_jobs is None else n_jobs)

    # generate html report (for QA) as desired
    parent_results_gallery = None
    if do_report:
        # what exactly was typed at the command-line (terminal) ?
        command_line = "python %s" % " ".join(sys.argv)

        # copy web stuff to output_dir
        copy_web_conf_files(output_dir)

        report_log_filename = os.path.join(
            output_dir, 'report_log.html')
        report_preproc_filename = os.path.join(
            output_dir, 'report_preproc.html')
        report_filename = os.path.join(
            output_dir, 'report.html')

        # get caller module handle from stack-frame
        user_script_name = sys.argv[0]
        user_source_code = get_module_source_code(
            user_script_name)

        # generate docstring for preproc tobe undergone
        preproc_undergone = ""
        preproc_undergone += generate_preproc_undergone_docstring(
            command_line=command_line,
            # prepreproc_undergone=prepreproc_undergone,
            # do_dartel=do_dartel,
            # **do_subject_preproc_kwargs
            )

        # scrape this function's arguments
        preproc_params = ""
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        preproc_func_name = inspect.getframeinfo(frame)[2]
        preproc_params += ("Function <i>%s(...)</i> was invoked by the script"
                           " <i>%s</i> with the following arguments:"
                           ) % (preproc_func_name, user_script_name)
        args_dict = dict((arg, values[arg]) for arg in args if not arg in [
                "dataset_description",
                "report_filename",
                "do_report",
                "do_cv_tc",
                "do_shutdown_reloaders",
                "subjects",
                # add other args to exclude below
                ])
        args_dict['output_dir'] = output_dir
        preproc_params += dict_to_html_ul(
            args_dict
            )

        # initialize results gallery
        loader_filename = os.path.join(
            output_dir, "results_loader.php")
        parent_results_gallery = ResultsGallery(
            loader_filename=loader_filename,
            refresh_timeout=30,
            )

        # initialize progress bar
        progress_logger = ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename,
                                 report_preproc_filename])

        # html markup
        log = get_dataset_report_log_html_template(
            ).substitute(
            start_time=time.ctime(),
            )

        preproc = get_dataset_report_preproc_html_template(
            ).substitute(
            results=parent_results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            dataset_description=dataset_description,
            source_code=user_source_code,
            source_script_name=user_script_name,
            preproc_params=preproc_params
            )

        main_html = get_dataset_report_html_template(
            ).substitute(
            results=parent_results_gallery,
            start_time=time.ctime(),
            dataset_id=dataset_id,
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

        if not do_dartel:
            do_subject_preproc_kwargs['parent_results_gallery'
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
        if not do_report:
            return

        progress_logger.finish(report_preproc_filename)

        if shutdown_reloaders:
            print "Finishing %s..." % output_dir
            progress_logger.finish_dir(output_dir)

        print "\r\n\tHTML report written to %s" % report_preproc_filename

    # preprocess subject's separately
    if do_dartel:
        fwhm = do_subject_preproc_kwargs.get("fwhm", 0.)
        do_subject_preproc_kwargs["fwhm"] = 0.
        do_cv_tc = do_subject_preproc_kwargs.get("do_cv_tc", True)
        for item in ["do_segment", "do_normalize", "do_cv_tc",
                     "last_stage"]:
            do_subject_preproc_kwargs[item] = False

    preproc_subject_data = Parallel(n_jobs=n_jobs)(
        delayed(do_subject_preproc)(
            subject_data, **do_subject_preproc_kwargs
            ) for subject_data in subjects)

    # DARTEL
    if do_dartel:
        preproc_subject_data = _do_subjects_dartel(
            preproc_subject_data, output_dir,
            n_jobs=n_jobs,
            fwhm=fwhm,
            do_report=do_subject_preproc_kwargs.get("do_report", True),
            do_cv_tc=do_cv_tc,
            parent_results_gallery=parent_results_gallery
            )

    finalize_report()

    return preproc_subject_data
