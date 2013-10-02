"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

"""

# standard imports
import numpy as np
import nibabel
import os
import sys
import shutil
import glob
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
import nipype.interfaces.matlab as matlab

# import API for i/o
from .io_utils import (load_specific_vol,
                       ravel_filenames,
                       unravel_filenames,
                       get_vox_dims,
                       hard_link
                       )

# import API for reporting
from .reporting.base_reporter import (Thumbnail,
                                      ResultsGallery,
                                      ProgressReport,
                                      a,
                                      img,
                                      copy_web_conf_files,
                                      get_subject_report_html_template,
                                      get_subject_report_preproc_html_template,
                                      get_dataset_report_preproc_html_template,
                                      get_dataset_report_html_template,
                                      get_dataset_report_log_html_template,
                                      dict_to_html_ul,
                                      get_module_source_code,
                                      commit_subject_thumnbail_to_parent_gallery
                                      )
from .reporting.preproc_reporter import (generate_preproc_undergone_docstring,
                                         generate_realignment_thumbnails,
                                         generate_coregistration_thumbnails,
                                         generate_segmentation_thumbnails,
                                         generate_normalization_thumbnails,
                                         generate_cv_tc_thumbnail,
                                         make_nipype_execution_log_html
                                         )

# configure MATLAB exec
MATLAB_EXEC = "/neurospin/local/matlab/bin/matlab"
if not os.path.exists(MATLAB_EXEC):
    m_choices = glob.glob("/neurospin/local/matlab/R*/bin/matlab")
    if m_choices:
        MATLAB_EXEC = m_choices[0]
if 'MATLAB_EXEC' in os.environ:
    MATLAB_EXEC = os.environ['MATLAB_EXEC']
assert os.path.exists(MATLAB_EXEC), \
    "nipype_preproc_smp_utils: MATLAB_EXEC: %s, \
doesn't exist; you need to export MATLAB_EXEC" % MATLAB_EXEC
matlab.MatlabCommand.set_default_matlab_cmd(MATLAB_EXEC)

# configure matlab SPM
SPM_DIR = '/i2bm/local/spm8'
if 'SPM_DIR' in os.environ:
    SPM_DIR = os.environ['SPM_DIR']
assert os.path.exists(SPM_DIR), \
    "nipype_preproc_smp_utils: SPM_DIR: %s,\
 doesn't exist; you need to export SPM_DIR" % SPM_DIR
matlab.MatlabCommand.set_default_paths(SPM_DIR)

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


def niigz2nii(ifilename, output_dir=None):
    """
    Converts .nii.gz to .nii (SPM doesn't support .nii.gz images).

    Parameters
    ----------
    ifilename: string, of list of strings
        input filename of image to be extracted

    output_dir: string, optional (default None)
        output directory to which exctracted file will be written.
        If no value is given, the output will be written in the parent
        directory of ifilename

    Returns
    -------
    ofilename: string
        filename of extracted image

    """

    if isinstance(ifilename, list):
        return [niigz2nii(x, output_dir=output_dir) for x in ifilename]
    else:
        assert isinstance(ifilename, basestring), (
            "ifilename must be string or list of strings, got %s" % type(
                ifilename))

    if not ifilename.endswith('.nii.gz'):
        return ifilename

    ofilename = ifilename[:-3]
    if not output_dir is None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ofilename = os.path.join(output_dir, os.path.basename(ofilename))

    nibabel.save(nibabel.load(ifilename), ofilename)

    return ofilename


class SubjectData(object):
    """
    Encapsulation for subject data, relative to preprocessing.

    Parameters
    ----------
    func: can be one of the following types:
        ---------------------------------------------------------------
        Type                       | Explanation
        ---------------------------------------------------------------
        string                     | one session, 1 4D image filename
        ---------------------------------------------------------------
        list of strings            | one session, multiple 3D image
                                   | filenames (one per scan)
                                   | OR multiple sessions, multiple 4D
                                   | image filenames (one per session)
        ---------------------------------------------------------------
        list of list of strings    | multiiple sessions, one list of
                                   | 3D image filenames (one per scan)
                                   | per session
        ---------------------------------------------------------------

    anat: string
        path to anatomical image

    subject_id: string, optional (default 'sub001')
        subject id

    session_id: string or list of strings, optional (default None):
        session ids for all sessions (i.e runs)

    """

    def __init__(self, func=None, anat=None, subject_id="sub001",
                 session_id=None, output_dir=None):
        self.func = func
        self.anat = anat
        self.subject_id = subject_id
        self.session_id = session_id
        self.output_dir = output_dir

    def sanitize(self):
        """
        This method does basic sanitization of the `SubjectData` instance, like
        extracting .nii.gz -> .nii (crusial for SPM), ensuring that functional
        images actually exist on disk, etc.

        """

        # sanitize output_dir
        assert not self.output_dir is None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # sanitize func
        if isinstance(self.func, basestring):
            self.func = [self.func]
        self.func = niigz2nii(self.func, output_dir=self.output_dir)

        # sanitize anat
        if not self.anat is None:
            self.anat = niigz2nii(self.anat, output_dir=self.output_dir)

        # sanitize session_id
        if self.session_id is None:
            self.session_id = ["session_%i" % i
                               for i in xrange(len(self.func))]
        else:
            if isinstance(self.session_id, basestring):
                assert len(self.func) == 1
                self.session_id = [self.session_id]
            else:
                assert len(self.session_id) == len(self.func), "%s != %s" % (
                    self.session_id, len(self.func))

    def hard_link_output_files(self):
        """
        Hard-links output files to subject's immediate output directory.

        """

        for item in ['func', 'anat', 'realignment_parameters',
                     'gm', 'wm', 'csf',
                     'wgm', 'wwm', 'wcsf'
                     ]:
            if hasattr(self, item):
                setattr(self, item, hard_link(getattr(self, item),
                                              self.output_dir))


def _do_subject_realign(subject_data, nipype_mem=None,
                        do_report=True, results_gallery=None,
                        ignore_exception=False):
    """
    Wrapper for running spm.Realign with optional reporting.

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functional images (subject_data.func) are to be
        realigned

    nipype_mem: `nipype.caching.Memory` object
        Wrapper for running node with nipype.caching.Memory

    do_report: bool, optional (default True)
       flag controlling whether post-preprocessing reports should be generated

    results_gallery: `ResultsGallery` object
       initialized results gallery for the subject being preprocessed; new QA
       thumbnails will be committed here

    ignore_exception: bool, optional (default False)
       parameter passed to spm.Realign class, controls what should be done
       in case an exception prevails when the node is executed

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
        register_to_mean=True,
        ignore_exception=ignore_exception
        )

    # collect output
    subject_data.nipype_results['realign'] = realign_result
    subject_data.func = realign_result.outputs.realigned_files
    subject_data.realignment_parameters = \
        realign_result.outputs.realignment_parameters

    # generate report
    if do_report:
        # generate realignment thumbs
        thumbs = generate_realignment_thumbnails(
            subject_data.realignment_parameters,
            subject_data.output_dir,
            sessions=subject_data.session_id,
            execution_log_html_filename=make_nipype_execution_log_html(
                realign_result.outputs.realigned_files, "Realign",
                subject_data.output_dir),
            results_gallery=results_gallery
            )

        subject_data.final_thumbnail.img.src = thumbs['rp_plot']

    return subject_data


def _do_subject_coregister(subject_data, coreg_anat_to_func=False,
                           nipype_mem=None, joblib_mem=None,
                           ignore_exception=False,
                           do_report=True, results_gallery=None
                           ):
    """
    Wrapper for running spm.Coregister with optional reporting.

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

    results_gallery: `ResultsGallery` object
       initialized results gallery for the subject being preprocessed; new QA
       thumbnails will be committed here

    ignore_exception: bool, optional (default False)
       parameter passed to spm.Coregister class, controls what should be done
       in case an exception prevails when the node is executed

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

            nipype_mem = NipypeMemory(base_dir=cache_dir)

    def _save_vol(data, affine, output_filename):
        nibabel.save(nibabel.Nifti1Image(data, affine), output_filename)

    # config node
    coreg = nipype_mem.cache(spm.Coregister)
    coreg_jobtype = 'estimate'
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
        coreg_source = os.path.join(subject_data.output_dir,
                                "ref_func_vol.nii")
        joblib_cached(_save_vol)(ref_func.get_data(),
                                 ref_func.get_affine(),
                                 coreg_source)

        apply_to_files, file_types = ravel_filenames(subject_data.func)

    # run node
    coreg_result = coreg(target=coreg_target,
                         source=coreg_source,
                         apply_to_files=apply_to_files,
                         jobtype=coreg_jobtype,
                         ignore_exception=ignore_exception
                         )

    # collect output
    subject_data.nipype_results['coreg'] = coreg_result
    if coreg_anat_to_func:
        subject_data.anat = coreg_result.outputs.coregistered_source
    else:
        subject_data.func = unravel_filenames(
            coreg_result.outputs.coregistered_files, file_types)

    # report coreg
    if do_report:
        thumbs = generate_coregistration_thumbnails(
            (coreg_target, ref_brain),
            (coreg_result.outputs.coregistered_source, src_brain),
            subject_data.output_dir,
            execution_log_html_filename=make_nipype_execution_log_html(
                coreg_result.outputs.coregistered_source, "Coregister",
                subject_data.output_dir),
            results_gallery=results_gallery
            )

        subject_data.final_thumbnail.img.src = thumbs['axial']

    return subject_data


def _do_subject_segment(subject_data, do_normalize=False, nipype_mem=None,
                        do_report=True, results_gallery=None,
                        ignore_exception=False):
    """
    Wrapper for running spm.Segment with optional reporting.

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

    results_gallery: `ResultsGallery` object
       initialized results gallery for the subject being preprocessed; new QA
       thumbnails will be committed here

    ignore_exception: bool, optional (default False)
       parameter passed to spm.Segment class, controls what should be done
       in case an exception prevails when the node is executed

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
        ignore_exception=ignore_exception
        )

    # collect output
    subject_data.nipype_results['segment'] = segment_result
    subject_data.gm = segment_result.outputs.native_gm_image
    subject_data.wm = segment_result.outputs.native_wm_image
    subject_data.csf = segment_result.outputs.native_csf_image
    if do_normalize:
        subject_data.wgm = segment_result.outputs.normalized_gm_image
        subject_data.wwm = segment_result.outputs.normalized_wm_image
        subject_data.wcsf = segment_result.outputs.normalized_csf_image

    # do report
    if do_report:
        for brain_name, brain, cmap in zip(
            ['anat', 'func'], [subject_data.anat, subject_data.func],
            [cm.gray, cm.spectral]):
            thumbs = generate_segmentation_thumbnails(
                subject_data.anat,
                subject_data.output_dir,
                subject_gm_file=subject_data.gm,
                subject_wm_file=subject_data.wm,
                subject_csf_file=subject_data.csf,
                cmap=cmap,
                brain=brain_name,
                only_native=True,
                execution_log_html_filename=make_nipype_execution_log_html(
                    segment_result.outputs.transformation_mat, "Segment",
                    subject_data.output_dir),
                results_gallery=results_gallery
                )

            if brain_name == 'func':
                subject_data.final_thumbnail.img.src = thumbs['axial']

    return subject_data


def _do_subject_normalize(subject_data, nipype_mem, fwhm=0., do_report=True,
                          results_gallery=None, ignore_exception=False):
    """
    Wrapper for running spm.Segment with optional reporting.

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

    results_gallery: `ResultsGallery` object
       initialized results gallery for the subject being preprocessed; new QA
       thumbnails will be committed here

    ignore_exception: bool, optional (default False)
       parameter passed to spm.Normalize class, controls what should be done
       in case an exception prevails when the node is executed

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
    Input subject_data is modified.

    """

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
            else:
                apply_to_files = subject_data.anat

            # run node
            normalize_result = normalize(
                parameter_file=parameter_file,
                apply_to_files=apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=get_vox_dims(apply_to_files),
                write_interp=1,
                jobtype='write',
                ignore_exception=ignore_exception
                )
        else:
            if brain_name == 'func':
                apply_to_files, file_types = ravel_filenames(
                    subject_data.func)
            else:
                apply_to_files = subject_data.anat

            # warp brain
            normalize_result = normalize(
                parameter_file=parameter_file,
                apply_to_files=apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=get_vox_dims(apply_to_files),
                write_wrap=[0, 0, 0],
                write_interp=1,
                jobtype='write',
                ignore_exception=ignore_exception
                )

            subject_data.nipype_results['normalize_%s' % brain_name
                                        ] = normalize_result
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

            # explicit smoothing
            if np.sum(fwhm) > 0:
                subject_data = _do_subject_smooth(
                    subject_data, fwhm, nipype_mem=nipype_mem,
                    ignore_exception=ignore_exception
                    )
        else:
            subject_data.anat = normalize_result.outputs.normalized_files

        # do report
        if do_report:
            # generate segmentation thumbs
            if segmented:
                thumbs = generate_segmentation_thumbnails(
                    subject_data.anat,
                    subject_data.output_dir,
                    subject_gm_file=subject_data.wgm,
                    subject_wm_file=subject_data.wwm,
                    subject_csf_file=subject_data.wcsf,
                    cmap=cmap,
                    brain=brain_name,
                    comments="warped",
                    execution_log_html_filename=make_nipype_execution_log_html(
                        subject_data.nipype_results[
                            "segment"].outputs.transformation_mat, "Segment",
                        subject_data.output_dir),
                    results_gallery=results_gallery
                    )

                if brain_name == 'func':
                    subject_data.final_thumbnail.img.src = thumbs['axial']

            # generate normalization thumbs
            generate_normalization_thumbnails(
                normalize_result.outputs.normalized_files,
                subject_data.output_dir,
                brain=brain_name,
                execution_log_html_filename=make_nipype_execution_log_html(
                    normalize_result.outputs.normalized_files, "Normalize",
                    subject_data.output_dir),
                results_gallery=results_gallery,
                )

            if not segmented and brain_name == 'func':
                subject_data.final_thumbnail.img.src = thumbs['axial']

    return subject_data


def _do_subject_smooth(subject_data, fwhm, nipype_mem=None,
                       ignore_exception=False):
    """
    Wrapper for running spm.Smooth with optional reporting.

    Parameters
    -----------
    subject_data: `SubjectData` object
        subject data whose functional images (subject_data.func) are to be
        smoothed

    nipype_mem: `nipype.caching.Memory` object
        Wrapper for running node with nipype.caching.Memory

    results_gallery: `ResultsGallery` object
       initialized results gallery for the subject being preprocessed; new QA
       thumbnails will be committed here

    ignore_exception: bool, optional (default False)
       parameter passed to spm.Smooth class, controls what should be done
       in case an exception prevails when the node is executed

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

    # prepare for smart caching
    if nipype_mem is None:
        cache_dir = os.path.join(subject_data.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        nipype_mem = NipypeMemory(base_dir=cache_dir)

    smooth = nipype_mem.cache(spm.Smooth)
    in_files, file_types = ravel_filenames(subject_data.func)
    smooth_result = smooth(in_files=in_files,
                           fwhm=fwhm,
                           ignore_exception=ignore_exception
                           )

    # collect results
    subject_data.nipype_results['smooth'] = smooth_result
    subject_data.func = unravel_filenames(
        smooth_result.outputs.smoothed_files, file_types)

    return subject_data


def do_subject_preproc(
    subject_data,
    do_realign=True,
    do_coreg=True,
    coreg_anat_to_func=False,
    do_segment=True,
    do_normalize=True,
    do_cv_tc=True,
    fwhm=0.,
    hard_link_output=True,
    do_report=True,
    parent_results_gallery=None,
    last_stage=True,
    ignore_exception=False
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

    hard_link_output: bool, optional (default True)
        if set, then output files will be hard-linked from the respective
        nipype cache directories, to the subject's immediate output directory
        (subject_data.output_dir)

    do_cv_tc: bool (optional)
        if set, a summarizing the time-course of the coefficient of variation
        in the preprocessed fMRI time-series will be generated

    """

    # # print input args
    # frame = inspect.currentframe()
    # args, _, _, values = inspect.getargvalues(frame)
    # print "\r\n"
    # for i in args:
    #     print "\t %s=%s" % (i, values[i])
    # print "\r\n"

    # sanitze subject data
    subject_data.sanitize()

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
    results_gallery = None
    if do_report:
        # generate explanation of preproc steps undergone by subject
        preproc_undergone = generate_preproc_undergone_docstring(
            do_realign=do_realign,
            do_coreg=do_coreg,
            do_segment=do_segment,
            do_normalize=do_normalize,
            fwhm=fwhm,
            coreg_func_to_anat=not coreg_anat_to_func
            )

        # report filenames
        report_log_filename = os.path.join(subject_data.output_dir,
                                           'report_log.html')
        report_preproc_filename = os.path.join(subject_data.output_dir,
                                               'report_preproc.html')
        report_filename = os.path.join(subject_data.output_dir,
                                       'report.html')

        # initialize results gallery
        loader_filename = os.path.join(subject_data.output_dir,
                                       "results_loader.php")
        results_gallery = ResultsGallery(
            loader_filename=loader_filename,
            title="Report for subject %s" % subject_data.subject_id)

        # final thumbnail most representative of this subject's QA
        subject_data.final_thumbnail = Thumbnail()
        subject_data.final_thumbnail.a = a(href=report_preproc_filename)
        subject_data.final_thumbnail.img = img(src=None)
        subject_data.final_thumbnail.description = subject_data.subject_id

        # copy web stuff to subject output dir
        copy_web_conf_files(subject_data.output_dir)

        # initialize progress bar
        progress_logger = ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename,
                                 report_preproc_filename])
        subject_data.progress_logger = progress_logger

        # html markup
        preproc = get_subject_report_preproc_html_template(
            ).substitute(
            results=results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            subject_id=subject_data.subject_id,
            )
        main_html = get_subject_report_html_template(
            ).substitute(
            start_time=time.ctime(),
            subject_id=subject_data.subject_id
            )

        with open(report_preproc_filename, 'w') as fd:
            fd.write(str(preproc))
            fd.close()
        with open(report_filename, 'w') as fd:
            fd.write(str(main_html))
            fd.close()

        def _finalize_report():
            """
            Finalizes the business of reporting.

            """

            # generate CV thumbs
            if do_report or do_cv_tc:
                generate_cv_tc_thumbnail(subject_data.func,
                                         subject_data.session_id,
                                         subject_data.subject_id,
                                         subject_data.output_dir,
                                         results_gallery=results_gallery
                                         )

                progress_logger.finish(report_preproc_filename)

                if last_stage:
                    if not parent_results_gallery is None:
                        commit_subject_thumnbail_to_parent_gallery(
                            subject_data.final_thumbnail,
                            subject_data.subject_id,
                            parent_results_gallery)

                    progress_logger.finish_dir(subject_data.output_dir)

                print ("\r\nPreproc report for subject %s written to %s"
                       " .\r\n" % (subject_data.subject_id,
                                   report_preproc_filename))

    #######################
    #  motion correction
    #######################
    if do_realign:
        subject_data = _do_subject_realign(subject_data, nipype_mem=nipype_mem,
                                           do_report=do_report,
                                           results_gallery=results_gallery,
                                           ignore_exception=ignore_exception
                                           )

    ##################################################################
    # co-registration of structural (anatomical) against functional
    ##################################################################
    if do_coreg:
        subject_data = _do_subject_coregister(
            subject_data, nipype_mem=nipype_mem,
            joblib_mem=joblib_mem,
            coreg_anat_to_func=coreg_anat_to_func,
            do_report=do_report,
            results_gallery=results_gallery,
            ignore_exception=ignore_exception
            )

    #####################################
    # segmentation of anatomical image
    #####################################
    if do_segment:
        subject_data = _do_subject_segment(subject_data, nipype_mem=nipype_mem,
                                           do_normalize=do_normalize,
                                           do_report=do_report,
                                           results_gallery=results_gallery,
                                           ignore_exception=ignore_exception
                                           )

    ##########################
    # Spatial Normalization
    ##########################
    if do_normalize:
        subject_data = _do_subject_normalize(
            subject_data,
            fwhm,  # smooth func after normalization
            nipype_mem=nipype_mem,
            do_report=do_report,
            results_gallery=results_gallery,
            ignore_exception=ignore_exception
            )

    #########################################
    # Smooth without Spatial Normalization
    #########################################
    if not do_normalize and np.sum(fwhm) > 0:
        subject_data = _do_subject_smooth(subject_data, fwhm, nipype_mem=nipype_mem,
                                          ignore_exception=ignore_exception
                                          )

    # hard-link output files to subject's immediate output directory
    if hard_link_output:
        subject_data.hard_link_output_files()

    # finalize reports
    if do_report:
        _finalize_report()

    # return preprocessed subject_data
    return subject_data


def do_subjects_preproc(subject_factory,
                        output_dir=None,
                        n_jobs=None,
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

    n_jobs: int, optional (default None)
        number of jobs to create; parameter passed to `joblib.Parallel`.
        if N_JOBS is defined in the shell environment, then its value
        is used instead.

    do_subject_preproc_kwargs: parameter-value dict
        optional parameters passed to the \do_subject_preproc` API. See
        the API documentation for details

    Returns
    -------
    preproc_subject_data: list of preprocessed `SubjectData` objects
        each elements is the preprocessed version of the corresponding
        input `SubjectData` object

    """

    # sanitize output_dir
    if not output_dir is None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # configure number of jobs
    n_jobs = int(os.environ['N_JOBS']) if 'N_JOBS' in os.environ else (
        -1 if n_jobs is None else n_jobs)

    # get caller module handle from stack-frame
    user_script_name = sys.argv[0]
    user_source_code = get_module_source_code(
        user_script_name)

    preproc_undergone = ""

    preproc_undergone += generate_preproc_undergone_docstring(
        prepreproc_undergone=prepreproc_undergone,
        **do_subject_preproc_kwargs
        )

    # generate html report (for QA) as desired
    parent_results_gallery = None
    if do_report:
        copy_web_conf_files(output_dir)

        report_log_filename = os.path.join(
            output_dir, 'report_log.html')
        report_preproc_filename = os.path.join(
            output_dir, 'report_preproc.html')
        report_filename = os.path.join(
            output_dir, 'report.html')

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
                "do_export_report",
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

        def finalize_report():
            progress_logger.finish(report_preproc_filename)

            if shutdown_reloaders:
                print "Finishing %s..." % output_dir
                progress_logger.finish_dir(output_dir)

        do_subject_preproc_kwargs['parent_results_gallery'
                                  ] = parent_results_gallery

    preproc_subject_data = Parallel(n_jobs=n_jobs)(delayed(do_subject_preproc)(
            subject_data, **do_subject_preproc_kwargs
            ) for subject_data in subject_factory)

    return preproc_subject_data
