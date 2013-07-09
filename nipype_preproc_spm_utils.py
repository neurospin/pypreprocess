"""
Module: nipype_preproc_spm_utils
Synopsis: routine functions for SPM preprocessing business
Author: dohmatob elvis dopgima (hereafter referred to as DED)

XXX TODO: document the code!
XcXX TODO: re-factor the code!

"""

# standard imports
import os
import sys
import shutil
import json
import glob
import time
import inspect

# imports for caching (yeah, we aint got time to loose!)
from nipype.caching import Memory

# reporting imports
import reporting.preproc_reporter as preproc_reporter
import reporting.base_reporter as base_reporter

import reporting.check_preprocessing as check_preprocessing
import pylab as pl

# imports i/o
import numpy as np
import nibabel
from nipype.interfaces.base import Bunch
from io_utils import delete_orientation, is_3D, get_vox_dims,\
    resample_img, do_3Dto4D_merge, compute_mean_image
from datasets_extras import unzip_nii_gz

# spm and matlab imports
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab

# parallelism imports
import joblib

# set job count
N_JOBS = -1
if 'N_JOBS' in os.environ:
    N_JOBS = int(os.environ['N_JOBS'])

# set matlab exec path
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

# set matlab SPM back-end path
SPM_DIR = '/i2bm/local/spm8'
if 'SPM_DIR' in os.environ:
    SPM_DIR = os.environ['SPM_DIR']
assert os.path.exists(SPM_DIR), \
    "nipype_preproc_smp_utils: SPM_DIR: %s,\
 doesn't exist; you need to export SPM_DIR" % SPM_DIR
matlab.MatlabCommand.set_default_paths(SPM_DIR)

# set templates
EPI_TEMPLATE = os.path.join(SPM_DIR, 'templates/EPI.nii')
T1_TEMPLATE = "/usr/share/data/fsl-mni152-templates/avg152T1.nii"
if not os.path.isfile(T1_TEMPLATE):
    T1_TEMPLATE += '.gz'
    if not os.path.exists(T1_TEMPLATE):
        T1_TEMPLATE = os.path.join(SPM_DIR, "templates/T1.nii")

#os.path.join(SPM_DIR, 'templates/T1.nii')
GM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/grey.nii')
WM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/white.nii')
CSF_TEMPLATE = os.path.join(SPM_DIR, 'tpm/csf.nii')

# MISC
SPM8_URL = "http://www.fil.ion.ucl.ac.uk/spm/software/spm8/"
PYPREPROCESS_URL = "https://github.com/neurospin/pypreprocess"
DARTEL_URL = ("http://www.fil.ion.ucl.ac.uk/spm/software/spm8/"
              "SPM8_Release_Notes.pdf")
NIPYPE_URL = "http://nipy.sourceforge.net/nipype/"


class SubjectData(Bunch):
    """
    Encapsulation for subject data, relative to preprocessing.

    XXX Use custom objects (dicts, tuples, etc.) instead of this 'Bunch' stuff.

    """

    def __init__(self):
        """
        Data member, func, can be one of the following types:
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

        """

        Bunch.__init__(self)

        self.subject_id = "subXYZ"
        self.session_id = ["UNKNOWN_SESSION"]
        self.anat = None
        self.func = None
        self.hires = None
        self.bad_orientation = False
        self.output_dir = None

    def delete_orientation(self):
        # prepare for smart caching
        cache_dir = os.path.join(self.output_dir,
                                 'deleteorient_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        mem = joblib.Memory(cachedir=cache_dir, verbose=5)

        # deleteorient for func
        self.func = [mem.cache(delete_orientation)(
                self.func[j],
                self.output_dir,
                output_tag=self.session_id[j])
                     for j in xrange(len(self.session_id))]

        # deleteorient for anat
        if not self.anat is None:
            self.anat = mem.cache(delete_orientation)(
                self.anat, self.output_dir)

    def sanitize(self, do_deleteorient=False):
        if isinstance(self.session_id,
                      basestring) or isinstance(self.session_id, int):
            self.session_id = [self.session_id]

        if isinstance(self.func, basestring):
            self.func = [self.func]

        if isinstance(self.func[0], basestring):
            if is_3D(self.func[0]):
                self.func = [self.func]

        assert len(self.func) == len(self.session_id)

        if do_deleteorient or self.bad_orientation:
            self.delete_orientation()


def _do_subject_realign(output_dir,
                        sessions=None,
                        do_report=True,
                        results_gallery=None,
                        progress_logger=None,
                       **spm_realign_kwargs):
    """
    Wrapper for nipype.interfaces.spm.Realign.

    Does realignment and generates QA plots (motion parameters, etc.).

    Parameters
    ----------
    output_dir: string
        An existing folder where all output files will be written.

    subject_id: string (optional)
        id of the subject being preprocessed

    do_report: boolean (optional)
        if true, then QA plots will be generated after executing the realign
        node.

    *spm_realign_kwargs: kwargs (paramete-value dict)
        parameters to be passed to the nipype.interfaces.spm.Realign back-end
        node

    """

    output = {}

    if progress_logger:
        progress_logger.log('<b>Motion Correction</b><br/><br/>')

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)
    # run workflow
    realign = mem.cache(spm.Realign)
    realign_result = realign(**spm_realign_kwargs)

    # generate gallery for HTML report
    if do_report:
        if not realign_result.outputs is None:
            estimated_motion = realign_result.outputs.realignment_parameters
            if isinstance(estimated_motion, basestring):
                estimated_motion = [estimated_motion]

            assert len(sessions) == len(estimated_motion), estimated_motion

            execution_log = preproc_reporter.get_nipype_report(
                preproc_reporter.get_nipype_report_filename(estimated_motion))
            execution_log_html_filename = os.path.join(
                output_dir,
                'realignment_execution_log.html'
                )

            open(execution_log_html_filename, 'w').write(
                execution_log)

            if progress_logger:
                progress_logger.log(
                    '<b>Realignment</b><br/><br/>')
                progress_logger.log(execution_log)
                progress_logger.log('<hr/>')

            output.update(preproc_reporter.generate_realignment_thumbnails(
                    estimated_motion,
                    output_dir,
                    sessions=sessions,
                    execution_log_html_filename=execution_log_html_filename,
                    results_gallery=results_gallery,
                    progress_logger=progress_logger,
                    ))

    # collect ouput
    output['result'] = realign_result

    return output


def _do_subject_coreg(output_dir,
                      subject_id=None,
                      do_report=True,
                      results_gallery=None,
                      progress_logger=None,
                      coreg_func_to_anat=False,
                      comments="",
                      **spm_coregister_kwargs):
    """
    Wrapper for nipype.interfaces.spm.Coregister.

    Does coregistration and generates QA plots (outline of coregistered source
    on target, etc.).

    Parameters
    ----------
    output_dir: string
        An existing folder where all output files will be written.

    subject_id: string (optional)
        id of the subject being preprocessed

    do_report: boolean (optional)
        if true, then QA plots will be generated after executing the coregister
        node.

    *spm_coregister_kwargs: kwargs (paramete-value dict)
        parameters to be passed to the nipype.interfaces.spm.Coregister
        back-end node

    """

    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    # run workflow
    coreg = mem.cache(spm.Coregister)
    coreg_result = coreg(**spm_coregister_kwargs)

    # generate gallery for HTML report
    if do_report:
        #
        # XXX move the following code to reporting.preproc_reporter.py
        #

        ref_brain = "EPI"
        source_brain = "anat"
        if coreg_func_to_anat:
            ref_brain, source_brain = source_brain, ref_brain

        if not coreg_result.outputs is None:
            target = spm_coregister_kwargs['target']
            source = coreg_result.outputs.coregistered_source

            execution_log = preproc_reporter.get_nipype_report(
                preproc_reporter.get_nipype_report_filename(source))
            execution_log_html_filename = os.path.join(
                output_dir,
                'coregistration_execution_log.html'
                )

            open(execution_log_html_filename, 'w').write(
                execution_log)

            if progress_logger:
                progress_logger.log(
                    '<b>Coregistration</b><br/><br/>')
                progress_logger.log(execution_log)
                progress_logger.log('<hr/>')

            output.update(preproc_reporter.generate_coregistration_thumbnails(
                (target, ref_brain),
                (source, source_brain),
                output_dir,
                execution_log_html_filename=execution_log_html_filename,
                results_gallery=results_gallery,
                progress_logger=progress_logger,
                ))

    # collect ouput
    output['result'] = coreg_result

    return output


def _do_subject_segment(output_dir,
                        subject_id=None,
                        do_report=True,
                        progress_logger=None,
                        **spm_segment_kwargs):
    """
    Wrapper for nipype.interfaces.spm.Segment.

    Does segmentation of brain into GM, WM, and CSF compartments and
    generates QA plots.

    Parameters
    ----------
    output_dir: string
        An existing folder where all output files will be written.

    subject_id: string (optional)
        id of the subject being preprocessed

    do_report: boolean (optional)
        if true, then QA plots will be generated after executing the segment
        node.

    *spm_segment_kwargs: kwargs (parameter-value dict)
        parameters to be passed to the nipype.interfaces.spm.Segment back-end
        node

    """

    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    # run workflow
    segment = mem.cache(spm.Segment)
    segment_result = segment(**spm_segment_kwargs)

    # generate gallery for HTML report
    if do_report:
        if not segment_result.outputs is None:
            # nipype report
            execution_log = preproc_reporter.get_nipype_report(
                preproc_reporter.get_nipype_report_filename(
                    segment_result.outputs.transformation_mat)
                )

            if progress_logger:
                progress_logger.log(
                    '<b>Segmentation</b><br/><br/>')
                progress_logger.log(execution_log)
                progress_logger.log('<hr/>')

    # collect ouput
    output['result'] = segment_result

    return output


def _do_subject_normalize(output_dir,
                          segment_result=None,
                          do_report=True,
                          results_gallery=None,
                          progress_logger=None,
                          brain="EPI",
                          cmap=None,
                          fwhm=0,
                          **spm_normalize_kwargs):
    """
    Wrapper for nipype.interfaces.spm.Normalize.

    Does normalization and generates QA plots (outlines of normalized files on
    template, etc.).

    Parameters
    ----------
    output_dir: string
        An existing folder where all output files will be written.

    subject_id: string (optional)
        id of the subject being preprocessed

    do_report: boolean (optional)
        if true, then QA plots will be generated after executing the normalize
        node.

    *spm_normalize_kwargs: kwargs (paramete-value dict)
        parameters to be passed to the nipype.interfaces.spm.Normalize back-end
        node

    """

    nipype_report_filenames = []

    # sanity
    def get_norm_apply_to_files(files):
        if isinstance(files, basestring):
            norm_apply_to_files = files
            file_types = 'string'
        else:
            file_types = []
            norm_apply_to_files = []
            for x in files:
                if isinstance(x, basestring):
                    norm_apply_to_files.append(x)
                    file_types.append('string')
                else:
                    norm_apply_to_files += x
                    file_types.append(('list', len(x)))

        return norm_apply_to_files, file_types

    if 'apply_to_files' in spm_normalize_kwargs:
        spm_normalize_kwargs['apply_to_files'], file_types = \
            get_norm_apply_to_files(spm_normalize_kwargs['apply_to_files'])

    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    # run workflow
    normalize = mem.cache(spm.Normalize)
    norm_result = normalize(**spm_normalize_kwargs)

    # collect ouput
    output['result'] = norm_result
    if not norm_result.outputs is None:
        normalized_files = norm_result.outputs.normalized_files

        # define execution log html output filename
        execution_log_html_filename = os.path.join(
            output_dir,
            'normalization_of_%s_execution_log.html' % brain,
            )

        # grab execution log
        execution_log = preproc_reporter.get_nipype_report(
            preproc_reporter.get_nipype_report_filename(normalized_files))

        # write execution log
        open(execution_log_html_filename, 'w').write(
            execution_log)

        # update progress bar
        if progress_logger:
            progress_logger.log(
                '<b>Normalization of %s</b><br/><br/>' % brain)
            progress_logger.log(execution_log)
            progress_logger.log('<hr/>')

        # do smoothing
        if np.sum(fwhm):
            smooth = mem.cache(spm.Smooth)
            smooth_result = smooth(
                in_files=output['result'].outputs.normalized_files,
                fwhm=fwhm)

            # collect ouput
            output['result'] = smooth_result
            if not smooth_result.outputs is None:
                normalized_files = smooth_result.outputs.smoothed_files
                output['normalized_files'] = normalized_files

                # grab execution log
                execution_log = preproc_reporter.get_nipype_report(
                    preproc_reporter.get_nipype_report_filename(
                        normalized_files))

                # write execution log
                open(execution_log_html_filename, 'w').write(
                    execution_log)

                # update progress bar
                if progress_logger:
                    progress_logger.log(
                        '<b>Smoothening of %s</b><br/><br/>' % brain)
                    progress_logger.log(execution_log)
                    progress_logger.log('<hr/>')

        if 'apply_to_files' in spm_normalize_kwargs:
            if not isinstance(file_types, basestring):
                _tmp = []
                s = 0
                for x in file_types:
                    if x == 'string':
                        if isinstance(normalized_files, basestring):
                            _tmp = normalized_files
                            break
                        else:
                            _tmp.append(
                                normalized_files[s])
                            s += 1
                    else:
                        _tmp.append(
                            normalized_files[s: s + x[1]])
                        s += x[1]

                normalized_files = _tmp

        output['normalized_files'] = normalized_files

    # generate gallery for HTML report
    if do_report:
        if normalized_files:
            if segment_result:
                subject_gm_file = segment_result.outputs.modulated_gm_image
                subject_wm_file = segment_result.outputs.modulated_wm_image
                subject_csf_file = segment_result.outputs.modulated_csf_image
            else:
                subject_gm_file = None
                subject_wm_file = None
                subject_csf_file = None

            nipype_report_filenames = [
                preproc_reporter.get_nipype_report_filename(
                    subject_gm_file)] + nipype_report_filenames

            # generate normalization thumbs
            output.update(preproc_reporter.generate_normalization_thumbnails(
                normalized_files,
                output_dir,
                brain=brain,
                execution_log_html_filename=execution_log_html_filename,
                results_gallery=results_gallery,
                ))

            # generate segmentation thumbs
            output.update(preproc_reporter.generate_segmentation_thumbnails(
                normalized_files,
                output_dir,
                subject_gm_file=subject_gm_file,
                subject_wm_file=subject_wm_file,
                subject_csf_file=subject_csf_file,
                brain=brain,
                execution_log_html_filename=execution_log_html_filename,
                cmap=cmap,
                results_gallery=results_gallery))

    # collect ouput
    output['result'] = norm_result

    return output


def _do_subject_smooth(output_dir,
                        do_report=True,
                        results_gallery=None,
                        progress_logger=None,
                        brain="EPI",
                        cmap=None,
                        **spm_smooth_kwargs):
    output = {}

    # sanity
    def get_norm_apply_to_files(files):
        if isinstance(files, basestring):
            norm_apply_to_files = files
            file_types = 'string'
        else:
            file_types = []
            norm_apply_to_files = []
            for x in files:
                if isinstance(x, basestring):
                    norm_apply_to_files.append(x)
                    file_types.append('string')
                else:
                    norm_apply_to_files += x
                    file_types.append(('list', len(x)))

        return norm_apply_to_files, file_types

    if 'in_files' in spm_smooth_kwargs:
        spm_smooth_kwargs['in_files'], file_types = \
            get_norm_apply_to_files(spm_smooth_kwargs['in_files'])

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    # do smoothing
    smooth = mem.cache(spm.Smooth)
    smooth_result = smooth(**spm_smooth_kwargs)

    # collect ouput
    output['result'] = smooth_result
    if not smooth_result.outputs is None:
        smoothed_files = smooth_result.outputs.smoothed_files
        output['smoothed_files'] = smoothed_files

        if 'in_files' in spm_smooth_kwargs:
            if not isinstance(file_types, basestring):
                _tmp = []
                s = 0
                for x in file_types:
                    if x == 'string':
                        if isinstance(smoothed_files, basestring):
                            _tmp = smoothed_files
                            break
                        else:
                            _tmp.append(
                                smoothed_files[s])
                            s += 1
                    else:
                        _tmp.append(
                            smoothed_files[s: s + x[1]])
                        s += x[1]

                smoothed_files = _tmp

        output['smoothed_files'] = smoothed_files

        # define execution log html output filename
        execution_log_html_filename = os.path.join(
            output_dir,
            'smoothening_of_%s_execution_log.html' % brain,
            )

        # grab execution log
        execution_log = preproc_reporter.get_nipype_report(
            preproc_reporter.get_nipype_report_filename(
                smoothed_files))

        # write execution log
        open(execution_log_html_filename, 'w').write(
            execution_log)

        # update progress bar
        if progress_logger:
            progress_logger.log(
                '<b>Smoothening of %s</b><br/><br/>' % brain)
            progress_logger.log(execution_log)
            progress_logger.log('<hr/>')

    # collect ouput
    output['result'] = smooth_result
    return output


def _do_subject_preproc(
    subject_data,
    do_deleteorient=False,
    do_report=True,
    fwhm=None,
    do_bet=False,
    do_slicetiming=False,
    slice_order='ascending',
    interleaved=False,
    do_realign=True,
    do_coreg=True,
    func_to_anat=False,
    do_segment=True,
    do_normalize=True,
    do_cv_tc=True,
    additional_preproc_undergone=None,
    parent_results_gallery=None,
    subject_progress_logger=None,
    last_stage=True,
    ignore_exception=True):
    """
    Function preprocessing data for a single subject.

    Parameters
    ----------
    subject_data: instance of SubjectData
        Object containing information about the subject under inspection
        (path to anat image, func image(s),
        output directory, etc.)

    delete_orientation: bool (optional)
        if true, then orientation meta-data in all input image files for this
        subject will be stripped-off

    do_report: bool (optional)
        if set, post-preprocessing QA report will be generated.

    do_bet: bool (optional)
        if set, brain-extraction will be applied to remove non-brain tissue
        before preprocessing (this can help prevent the scull from aligning
        with the cortical surface, for example)

    do_slicetiming: bool (optional)
        if set, slice-timing correct temporal mis-alignment of the functional
        slices

    do_realign: bool (optional)
        if set, then the functional data will be realigned to correct for
        head-motion

    do_coreg: bool (optional)
        if set, then subject anat image (of the spm EPI template, if the later
        if not available) will be coregistered against the functional images
        (i.e the mean thereof)

    do_segment: bool (optional)
        if set, then the subject's anat image will be segmented to produce GM,
        WM, and CSF compartments (useful for both indirect normalization
        (intra-subject) or DARTEL (inter-subject) alike


    do_cv_tc: bool (optional)
        if set, a summarizing the time-course of the coefficient of variation
        in the preprocessed fMRI time-series will be generated

    parent_results_gallery: base_reporter.ResulsGallery object (optional)
        a handle to the results gallery to which the final QA thumail for this
        subject will be committed

    """

    # sanity
    subject_data.sanitize(do_deleteorient=do_deleteorient)

    output = {"subject_id": subject_data.subject_id,
              "session_id": subject_data.session_id,
              "output_dir": subject_data.output_dir,
              "func": subject_data.func,
              "anat": subject_data.anat}

    if not fwhm is None:
        if not (isinstance(fwhm, tuple) or isinstance(fwhm, list)):
            fwhm = [fwhm] * 3
        fwhm = list(fwhm)

    # create subject_data.output_dir if dir doesn't exist
    if not os.path.exists(subject_data.output_dir):
        os.makedirs(subject_data.output_dir)

    # generate explanation of preproc steps undergone by subject
    preproc_undergone = preproc_reporter.\
        generate_preproc_undergone_docstring(
        do_deleteorient=do_deleteorient or subject_data.bad_orientation,
        fwhm=fwhm,
        do_bet=do_bet,
        do_slicetiming=do_slicetiming,
        do_realign=do_realign,
        do_coreg=do_coreg,
        coreg_func_to_anat=func_to_anat,
        do_segment=do_segment,
        do_normalize=do_normalize,
        additional_preproc_undergone=additional_preproc_undergone,
        )

    output['preproc_undergone'] = preproc_undergone

    if do_report:
        # copy css and js stuff to output dir
        for js_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                          "js/*.js")):
            shutil.copy(js_file, subject_data.output_dir)
        for css_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                          "css/*.css")):
            shutil.copy(css_file, subject_data.output_dir)
        for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                                "icons/*.gif")):
            shutil.copy(icon_file, subject_data.output_dir)
        for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                                "images/*.png")):
            shutil.copy(icon_file, subject_data.output_dir)
        for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                                "images/*.jpeg")):
            shutil.copy(icon_file, subject_data.output_dir)

        report_log_filename = os.path.join(
            subject_data.output_dir, 'report_log.html')
        report_preproc_filename = os.path.join(
            subject_data.output_dir, 'report_preproc.html')
        report_filename = os.path.join(
            subject_data.output_dir, 'report.html')

        final_thumbnail = base_reporter.Thumbnail()
        final_thumbnail.a = base_reporter.a(href=report_preproc_filename)
        final_thumbnail.img = base_reporter.img(src=None)
        final_thumbnail.description = subject_data.subject_id

        # initialize results gallery
        loader_filename = os.path.join(
            subject_data.output_dir, "results_loader.php")
        results_gallery = base_reporter.ResultsGallery(
            loader_filename=loader_filename,
            title="Report for subject %s" % subject_data.subject_id)
        output['results_gallery'] = results_gallery

        # initialize progress bar
        if subject_progress_logger is None:
            subject_progress_logger = base_reporter.ProgressReport(
                report_log_filename,
                other_watched_files=[report_filename,
                                     report_preproc_filename,
                                     report_log_filename])
        output['progress_logger'] = subject_progress_logger

        # html markup
        log = base_reporter.get_subject_report_log_html_template(
            ).substitute(
            start_time=time.ctime(),
            subject_id=subject_data.subject_id
            )

        preproc = base_reporter.get_subject_report_preproc_html_template(
            ).substitute(
            results=results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            subject_id=subject_data.subject_id
            )

        main_html = base_reporter.get_subject_report_html_template(
            ).substitute(
            start_time=time.ctime(),
            subject_id=subject_data.subject_id
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
            output['final_thumbnail'] = final_thumbnail

            if last_stage:
                if parent_results_gallery:
                    base_reporter.commit_subject_thumnbail_to_parent_gallery(
                        final_thumbnail,
                        subject_data.subject_id,
                        parent_results_gallery)

                subject_progress_logger.finish(report_preproc_filename)
    else:
        results_gallery = None

    # brain extraction (bet)
    if do_bet:
        pass

    ###############
    # slice-timing
    ###############
    if do_slicetiming:
        raise NotImplementedError(
            "STC module not yet integrated into this pipeline.")

        # import algorithms.slice_timing.spm_slice_timing as spm_slice_timing

        # # st_cache_dir = os.path.join(subject_data.output_dir,
        # #                                      "cache_dir", "slice_timing")
        # # if not os.path.isdir(st_cache_dir):
        # #     os.makedirs(st_cache_dir)

        # # mem = joblib.Memory(cachedir=st_cache_dir, verbose=100)

        # def _load_fmri_data(fmri_files):
        #     """Helper function to load fmri data from filename /
        #     ndarray or list of such

        #     """

        #     if isinstance(fmri_files, np.ndarray):
        #         return fmri_files

        #     if isinstance(fmri_files, basestring):
        #         return nibabel.load(fmri_files).get_data()
        #     else:
        #         n_scans = len(fmri_files)
        #         _first = _load_fmri_data(fmri_files[0])
        #         data = np.ndarray(tuple(list(_first.shape[:3]
        #                                      ) + [n_scans]))
        #         data[..., 0] = _first
        #         for scan in xrange(1, n_scans):
        #             data[..., scan] = _load_fmri_data(fmri_files[scan])

        #         return data

        # def _save_stc_output(output_data, output_dir,
        #                      input_filenames,
        #                      prefix='a'):

        #     print "Saving STC output to %s..." % output_dir

        #     print output_data.shape
        #     n_scans = output_data.shape[-1]

        #     # sanitize output_diir
        #     ref_filename = input_filenames if isinstance(
        #         input_filenames, basestring) else input_filenames[0]
        #     ref_file_basename = os.path.basename(ref_filename)
        #     if output_dir is None:
        #         output_dir = output_dir
        #     if output_dir is None:
        #         output_dir = os.path.dirname(ref_filename)
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)

        #     # save realigned files to disk
        #     if isinstance(input_filenames, basestring):
        #         affine = nibabel.load(input_filenames).get_affine()

        #         for t in xrange(n_scans):
        #             output_filename = os.path.join(output_dir,
        #                                            "%s%i%s" % (
        #                     prefix, t, ref_file_basename))

        #             nibabel.save(nibabel.Nifti1Image(output_data[..., t],
        #                                              affine),
        #                          output_filename)

        #         output_filenames = output_filename
        #     else:
        #         output_filenames = []
        #         for filename, t in zip(input_filenames, xrange(n_scans)):
        #             affine = nibabel.load(filename).get_affine()
        #             output_filename = os.path.join(output_dir,
        #                                            "%s%s" % (
        #                     prefix,
        #                     os.path.basename(filename)))

        #             nibabel.save(nibabel.Nifti1Image(output_data[..., t],
        #                                              affine),
        #                          output_filename)

        #             output_filenames.append(output_filename)

        #     return output_filenames

        # stc = spm_slice_timing.STC()

        # stc_func = []
        # for s, func in zip(subject_data.session_id, subject_data.func):
        #     print "\r\nLoading fmri data for session %s..." % s
        #     fmri_data = _load_fmri_data(func)
        #     stc.fit(raw_data=fmri_data, slice_order=slice_order,
        #             interleaved=interleaved,)

        #     stc.transform(fmri_data)

        #     stc_func.append(_save_stc_output(
        #             stc.get_last_output_data(),
        #             os.path.join(subject_data.output_dir,
        #                          "STC_session%s" % s),
        #             func))

        # subject_data.func = stc_func

        # # # realigner = mem.cache(st.do_slicetiming_and_motion_correction)

        # # # run realigment ( = slice timing + motion correction)
        # # realigned_func_files, rp_files = realigner(
        # #     subject_data.func,
        # #     subject_data.output_dir,
        # #     tr=TR, slice_order=slice_order,
        # #     time_interp=True)

        # # # collect outputs (pipeline-like)
        # # subject_data.func = realigned_func_files
        # # output['estimated_motion'] = rp_files
        # # output['realigned_func'] = realigned_func_files

        # # # generate gallery for HTML report
        # # if do_report:
        # #     sessions = subject_data.session_id
        # #     estimated_motion = rp_files
        # #     if isinstance(estimated_motion, basestring):
        # #         estimated_motion = [estimated_motion]

        # #     assert len(sessions) == len(estimated_motion), estimated_motion

        # #     output.update(preproc_reporter.generate_realignment_thumbnails(
        # #             estimated_motion,
        # #             subject_data.output_dir,
        # #             sessions=sessions,
        # #             results_gallery=results_gallery,
        # #             progress_logger=subject_progress_logger,
        # #             ))
    #####################
    #  motion correction
    #####################
    if do_realign:
        realign_output = _do_subject_realign(
            subject_data.output_dir,
            sessions=subject_data.session_id,
            do_report=do_report,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            in_files=subject_data.func,
            register_to_mean=True,
            jobtype='estwrite',
            # ignore_exception=ignore_exception,
            )

        # collect output
        realign_result = realign_output['result']

        # if failed to realign, return
        if realign_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed realignment)'
                finalize_report()
                raise RuntimeError(
                    ("spm.Realign failed for subject %s (outputs is None)!"
                     ) % subject_data.subject_id)

        subject_data.func = realign_result.outputs.realigned_files
        ref_func = realign_result.outputs.mean_image

        output['realign_result'] = realign_result
        output['estimated_motion'
               ] = realign_result.outputs.realignment_parameters
        output['func'] = realign_result.outputs.realigned_files
        output['realigned_func'] = realign_result.outputs.realigned_files

        # generate report stub
        if do_report:
            final_thumbnail.img.src = realign_output['rp_plot']
    else:
        # manually compute mean (along time axis) of fMRI images
        # XXX derive a more sensible path for the ref_func
        ref_func = os.path.join(
            subject_data.output_dir,
            'meanfunc.nii')

        if len(subject_data.session_id) > 1:
            compute_mean_image(subject_data.func[0],
                               output_filename=ref_func)
        else:
            if isinstance(subject_data.func[0], basestring):
                compute_mean_image(subject_data.func[0],
                                   output_filename=ref_func)
            else:
                ref_func = subject_data.func[0][0]

    ################################################################
    # co-registration of structural (anatomical) against functional
    ################################################################
    if do_coreg:
        # specify input files for coregistration
        comments = "anat -> epi"
        if func_to_anat:
            comments = 'epi -> anat'
            coreg_target = subject_data.anat
            coreg_source = ref_func
        else:
            coreg_target = ref_func
            coreg_jobtype = 'estimate'
            if subject_data.anat is None:
                if not subject_data.hires is None:
                    coreg_source = subject_data.hires
                else:
                    # XXX skip coregistration altogether!!!
                    subject_data.anat = os.path.join(
                        subject_data.output_dir,
                        os.path.basename(EPI_TEMPLATE).replace('.gz', ''))
                    if not os.path.exists(subject_data.anat):
                        shutil.copy(
                            EPI_TEMPLATE, subject_data.output_dir)
                        unzip_nii_gz(subject_data.output_dir)
                    coreg_source = subject_data.anat
                do_segment = False
            else:
                coreg_source = subject_data.anat

        # run coreg proper
        coreg_output = _do_subject_coreg(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=do_report,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            coreg_func_to_anat=func_to_anat,
            comments=comments,
            target=coreg_target,
            source=coreg_source,
            jobtype=coreg_jobtype,
            ignore_exception=ignore_exception
            )

        # collect results
        coreg_result = coreg_output['result']
        output['coreg_result'] = coreg_result

        # if failed to coregister, return
        if coreg_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed coregistration)'
                finalize_report()
            raise RuntimeError(
                ("spm.Coregister failed for subject %s!"
                 ) % subject_data.subject_id)

        output['coregistered_anat'] = coreg_result.outputs.coregistered_source

        # reset anat to coregistered version thereof
        subject_data.anat = coreg_result.outputs.coregistered_source

        # generate report stub
        if do_report:
            final_thumbnail.img.src = coreg_output['axial']

    ###################################
    # segmentation of anatomical image
    ###################################
    if do_segment:
        segment_data = subject_data.anat
        segment_output = _do_subject_segment(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=do_report,
            progress_logger=subject_progress_logger,
            data=segment_data,
            gm_output_type=[True, True, True],
            wm_output_type=[True, True, True],
            csf_output_type=[True, True, True],
            tissue_prob_maps=[GM_TEMPLATE,
                              WM_TEMPLATE, CSF_TEMPLATE],
            gaussians_per_class=[2, 2, 2, 4],
            affine_regularization="mni",
            bias_regularization=0.0001,
            bias_fwhm=60,
            warping_regularization=1,
            ignore_exception=ignore_exception
            )

        segment_result = segment_output['result']
        output['gm'] = segment_result.outputs.normalized_gm_image
        output['wm'] = segment_result.outputs.normalized_wm_image


        # if failed to segment, return
        if segment_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed segmentation)'
                finalize_report()
            raise RuntimeError(
                ("spm.Segment failed for subject %s!"
                 ) % subject_data.subject_id)

        # output['segment_result'] = segment_result

        if do_normalize:
            ##############################################################
            # indirect normalization: warp fMRI images int into MNI space
            # using the deformations learned by segmentation
            ##############################################################
            norm_parameter_file = segment_result.outputs.transformation_mat
            norm_apply_to_files = subject_data.func

            norm_output = _do_subject_normalize(
                subject_data.output_dir,
                segment_result=segment_result,
                do_report=do_report,
                results_gallery=results_gallery,
                progress_logger=subject_progress_logger,
                brain='EPI',
                cmap=pl.cm.spectral,
                fwhm=fwhm,
                parameter_file=norm_parameter_file,
                apply_to_files=norm_apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=get_vox_dims(norm_apply_to_files),
                write_interp=1,
                jobtype='write',
                ignore_exception=ignore_exception
                )

            norm_result = norm_output["result"]

            # if failed to normalize, return None
            if norm_result.outputs is None:
                if do_report:
                    final_thumbnail.img.src = 'failed.png'
                    final_thumbnail.description += ' (failed normalization)'
                    finalize_report()
                raise RuntimeError(
                    ("spm.Normalize failed (EPI) for subject %s")
                    % subject_data.subject_id)

            subject_data.func = norm_result.outputs.normalized_files
            output['func'] = norm_output['normalized_files']

            if do_report:
                final_thumbnail.img.src = norm_output['axial']

            #########################################################
            # indirect normalization: warp anat image into MNI space
            # using the deformations learned by segmentation
            #########################################################
            norm_parameter_file = segment_result.outputs.transformation_mat
            norm_apply_to_files = subject_data.anat

            norm_output = _do_subject_normalize(
                subject_data.output_dir,
                segment_result=segment_result,
                brain="anat",
                cmap=pl.cm.gray,
                do_report=do_report,
                results_gallery=results_gallery,
                progress_logger=subject_progress_logger,
                parameter_file=norm_parameter_file,
                apply_to_files=norm_apply_to_files,
                write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
                write_voxel_sizes=get_vox_dims(norm_apply_to_files),
                write_wrap=[0, 0, 0],
                write_interp=1,
                jobtype='write',
                ignore_exception=ignore_exception
                )

            norm_result = norm_output['result']

            # if failed to normalize, return None
            if norm_result.outputs is None:
                if do_report:
                    final_thumbnail.img.src = 'failed.png'
                    final_thumbnail.description += ' (failed normalization)'
                    finalize_report()
                raise RuntimeError(
                    ("spm.Normalize failed (anat) for subject %s")
                    % subject_data.subject_id)

            output['anat'] = norm_output['normalized_files']

    elif do_normalize:
        ############################################
        # learn T1 deformation without segmentation
        ############################################
        t1_template = os.path.join(
            subject_data.output_dir,
            os.path.basename(EPI_TEMPLATE).replace('.gz', ''))
        if not os.path.exists(t1_template):
            shutil.copy(
                EPI_TEMPLATE, subject_data.output_dir)
            unzip_nii_gz(subject_data.output_dir)

        norm_output = _do_subject_normalize(
            subject_data.output_dir,
            do_report=False,
            source=subject_data.anat,
            template=t1_template,
            )

        norm_result = norm_output['result']

        # if failed to normalize, return None
        if norm_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                finalize_report()
            raise RuntimeError(
                ("spm.Normalize failed for subject %s")
                % subject_data.subject_id)

        ####################################################
        # Warp EPI into MNI space using learned deformation
        ####################################################
        norm_parameter_file = norm_result.outputs.normalization_parameters
        norm_apply_to_files = subject_data.func

        norm_output = _do_subject_normalize(
            subject_data.output_dir,
            brain="EPI",
            cmap=pl.cm.spectral,
            fwhm=fwhm,
            do_report=do_report,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=get_vox_dims(norm_apply_to_files),
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write',
            ignore_exception=ignore_exception
            )

        norm_result = norm_output["result"]

        # if failed to normalize, return None
        if norm_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed normalization)'
                finalize_report()
            raise RuntimeError(
                ("spm.Normalize failed (EPI) for subject %s")
                % subject_data.subject_id)

        subject_data.func = norm_output['normalized_files']
        output['func'] = subject_data.func

        #####################################################
        # Warp anat into MNI space using learned deformation
        #####################################################
        norm_apply_to_files = subject_data.anat

        norm_output = _do_subject_normalize(
            subject_data.output_dir,
            brain="anat",
            cmap=pl.cm.gray,
            do_report=do_report,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=get_vox_dims(norm_apply_to_files),
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write',
            ignore_exception=ignore_exception
            )

        norm_result = norm_output['result']

        # if failed to normalize, return None
        if norm_result.outputs is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed normalization)'
                finalize_report()
            raise RuntimeError(
                ("spm.Normalize failed (anat) for subject %s")
                % subject_data.subject_id)

        output['anat'] = norm_output['normalized_files']
    elif np.sum(fwhm) > 0:
        # smooth func
        smooth_output = _do_subject_smooth(
            subject_data.output_dir,
            results_gallery=results_gallery,
            progress_logger=subject_progress_logger,
            in_files=subject_data.func,
            fwhm=fwhm)

        smooth_result = smooth_output['result']

        if smooth_result is None:
            if do_report:
                final_thumbnail.img.src = 'failed.png'
                final_thumbnail.description += ' (failed smoothing)'
                finalize_report()
            raise RuntimeError(
                ("spm.Smooth failed (EPI) for subject %s")
                % subject_data.subject_id)

        output['func'] = smooth_output['smoothed_files']

        # smooth anat
        if not subject_data.anat is None:
            smooth_output = _do_subject_smooth(
                subject_data.output_dir,
                results_gallery=results_gallery,
                progress_logger=subject_progress_logger,
                in_files=subject_data.anat,
                fwhm=fwhm)

            smooth_result = smooth_output['result']

            if smooth_result is None:
                if do_report:
                    final_thumbnail.img.src = 'failed.png'
                    final_thumbnail.description += ' (failed smoothing)'
                    finalize_report()
                raise RuntimeError(
                    ("spm.Smooth failed (anat) for subject %s")
                    % subject_data.subject_id)

        output['anat'] = smooth_output['smoothed_files']

    if do_report:
        # generate cv plots
        if do_cv_tc and do_normalize:
            corrected_FMRI = output['func']

            thumbnail = preproc_reporter.generate_cv_tc_thumbnail(
                corrected_FMRI,
                subject_data.session_id,
                subject_data.subject_id,
                subject_data.output_dir,
                results_gallery=results_gallery)

            if final_thumbnail.img.src is None:
                final_thumbnail = thumbnail

        finalize_report()

    return subject_data, output


def _do_subject_dartelnorm2mni(output_dir,
                               structural_file,
                               functional_file,
                               subject_gm_file=None,
                               subject_wm_file=None,
                               sessions=None,
                               subject_id=None,
                               downsample_func=True,
                               do_report=True,
                               do_cv_tc=True,
                               final_thumbnail=None,
                               results_gallery=None,
                               parent_results_gallery=None,
                               subject_progress_logger=None,
                               last_stage=True,
                               ignore_exception=True,
                               **dartelnorm2mni_kwargs):
    """
    Uses spm.DARTELNorm2MNI to warp subject brain into MNI space.

    Parameters
    ----------
    output_dir: string
        existing directory; results will be cache here

    **dartelnorm2mni_kargs: parameter-value list
        options to be passes to spm.DARTELNorm2MNI back-end

    """

    output = {"subject_id": subject_id,
              "output_dir": output_dir}

    fwhm = 0
    if 'fwhm' in dartelnorm2mni_kwargs:
        fwhm = dartelnorm2mni_kwargs['fwhm']

    output['progress_logger'] = subject_progress_logger

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    dartelnorm2mni = mem.cache(spm.DARTELNorm2MNI)

    # warp subject gm image (produced by Segment or NewSegment)
    # into MNI space
    if subject_gm_file:
        dartelnorm2mni_result = dartelnorm2mni(apply_to_files=subject_gm_file,
                                               **dartelnorm2mni_kwargs)
        subject_gm_file = dartelnorm2mni_result.outputs.normalized_files
    # warp subject wm image (produced by Segment or NewSegment)
    # into MNI space
    if subject_wm_file:
        dartelnorm2mni_result = dartelnorm2mni(apply_to_files=subject_wm_file,
                                               **dartelnorm2mni_kwargs)
        subject_wm_file = dartelnorm2mni_result.outputs.normalized_files

    # warp functional image into MNI space
    # functional_file = do_3Dto4D_merge(functional_file)
    createwarped = mem.cache(spm.CreateWarped)
    createwarped_result = createwarped(
        image_files=functional_file,
        flowfield_files=dartelnorm2mni_kwargs['flowfield_files'],
        ignore_exception=ignore_exception
        )

    # if node failed, return None
    if createwarped_result.outputs is None:
        raise RuntimeError(
            ("spm.CreateWarped failed for subject %s")
            % subject_id)

    warped_files = createwarped_result.outputs.warped_files

    execution_log_html_filename = os.path.join(
        output_dir,
        'normalization_of_epi_execution_log.html',
        )

    # grab execution log
    execution_log = preproc_reporter.get_nipype_report(
        preproc_reporter.get_nipype_report_filename(warped_files))

    # dump execution log unto html output file
    open(execution_log_html_filename, 'w').write(
        execution_log)

    # log progress
    if subject_progress_logger:
        subject_progress_logger.log(
            '<b>Normalization of EPI</b><br/><br/>')
        subject_progress_logger.log(execution_log)
        subject_progress_logger.log('<hr/>')

    # do smoothing
    if fwhm:
        smooth = mem.cache(spm.Smooth)
        smooth_result = smooth(
            in_files=warped_files,
            fwhm=fwhm)

        # collect ouput
        output['result'] = smooth_result
        if not smooth_result.outputs is None:
            warped_files = smooth_result.outputs.smoothed_files

    # # down-sample warped epi to save disk space ?
    # if downsample_func:
    #     if isinstance(warped_files, basestring):
    #         warped_files = [warped_files]

    #     resampled_warped_files = []
    #     for warped_file in warped_files:
    #         # warped_file = do_3Dto4D_merge(warped_file)

    #         # compute new vox dims to down-sample to
    #         new_vox_dims = (np.array(get_vox_dims(warped_file)) \
    #                         + np.array(get_vox_dims(functional_file))) / 2.0

    #         # down-sample proper
    #         resampled_warped_file = resample_img(
    #             warped_file, new_vox_dims)
    #         resampled_warped_files.append(resampled_warped_file)

    #     warped_files = resampled_warped_files

            # grab execution log
            execution_log = preproc_reporter.get_nipype_report(
                preproc_reporter.get_nipype_report_filename(
                    warped_files))

            # write execution log
            open(execution_log_html_filename, 'w').write(
                execution_log)

            # update progress bar
            if subject_progress_logger:
                subject_progress_logger.log(
                    '<b>Smoothening of EPI</b><br/><br/>')
                subject_progress_logger.log(execution_log)
                subject_progress_logger.log('<hr/>')

    # do_QA
    if do_report and results_gallery:
        # generate normalization thumbs for EPI
        preproc_reporter.generate_normalization_thumbnails(
            warped_files,
            output_dir,
            brain='EPI',
            execution_log_html_filename=execution_log_html_filename,
            results_gallery=results_gallery,
            )

        # generate segmentation thumbs for EPI
        epi_thumbs = preproc_reporter.generate_segmentation_thumbnails(
                warped_files,
                output_dir,
                subject_gm_file=subject_gm_file,
                subject_wm_file=subject_wm_file,
                brain='EPI',
                cmap=pl.cm.spectral,
                execution_log_html_filename=execution_log_html_filename,
                results_gallery=results_gallery,
                )

    # warp anat into MNI space
    dartelnorm2mni_result = dartelnorm2mni(
        apply_to_files=structural_file,
        ignore_exception=ignore_exception,
        **dartelnorm2mni_kwargs
        )

    # if node failed, return None
    if dartelnorm2mni_result.outputs is None:
        raise RuntimeError(
            ("spm.DartelNorm2MNI failed for subject %s")
            % subject_id)

    # define execution log html filename
    execution_log_html_filename = os.path.join(
        output_dir,
        'normalization_of_anat_execution_log.html',
        )

    # grab execution log
    execution_log = preproc_reporter.get_nipype_report(
        preproc_reporter.get_nipype_report_filename(
            dartelnorm2mni_result.outputs.normalized_files))

    # dump execution log unto html output file
    open(execution_log_html_filename, 'w').write(
        execution_log)

    # log progress
    if subject_progress_logger:
        subject_progress_logger.log(
            '<b>Normalization of anat</b><br/><br/>')
        subject_progress_logger.log(execution_log)
        subject_progress_logger.log('<hr/>')

    # do_QA
    if do_report and results_gallery:
        preproc_reporter.generate_normalization_thumbnails(
            dartelnorm2mni_result.outputs.normalized_files,
            output_dir,
            brain='anat',
            execution_log_html_filename=execution_log_html_filename,
            results_gallery=results_gallery,
            )

        # generate segmentation thumbs
        preproc_reporter.generate_segmentation_thumbnails(
            dartelnorm2mni_result.outputs.normalized_files,
            output_dir,
            subject_gm_file=subject_gm_file,
            subject_wm_file=subject_wm_file,
            brain='anat',
            execution_log_html_filename=execution_log_html_filename,
            results_gallery=results_gallery,
            )

        # finalize report
        if parent_results_gallery:
            final_thumbnail.img.src = epi_thumbs['axial']
            base_reporter.commit_subject_thumnbail_to_parent_gallery(
                final_thumbnail,
                subject_id,
                parent_results_gallery)

    if do_report:
        # generate cv plots
        if do_cv_tc:
            preproc_reporter.generate_cv_tc_thumbnail(
                createwarped_result.outputs.warped_files,
                sessions,
                subject_id,
                output_dir,
                results_gallery=results_gallery)

        # shutdown page reloader
        if last_stage:
            subject_progress_logger.finish_dir(output_dir)

    # collect results and return
    output['func'] = warped_files
    output['anat'] = dartelnorm2mni_result.outputs.normalized_files
    output['gm'] = subject_gm_file
    output['wm'] = subject_wm_file

    output['results_gallery'] = results_gallery
    output['progress_logger'] = subject_progress_logger

    return output


def do_group_DARTEL(output_dir,
                    subject_ids,
                    session_ids,
                    structural_files,
                    functional_files,
                    fwhm=0,
                    subject_output_dirs=None,
                    do_report=False,
                    do_cv_tc=True,
                    subject_final_thumbs=None,
                    subject_results_galleries=None,
                    subject_progress_loggers=None,
                    parent_results_gallery=None,
                    ignore_exception=True):
    """
    Undocumented API!

    """

    if subject_results_galleries is None:
        subject_results_galleries = [None] * len(structural_files)

    if subject_progress_loggers is None:
        subject_progress_loggers = [None] * len(structural_files)

    if subject_final_thumbs is None:
        subject_final_thumbs = [None] * len(structural_files)

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

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
        channel_files=structural_files,
        tissues=[tissue1, tissue2, tissue3, tissue4, tissue5, tissue6],
        ignore_exception=ignore_exception
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

    # warp individual brains into group (DARTEL) space
    subject_gm_files = newsegment_result.outputs.dartel_input_images[
        0]
    subject_wm_files = newsegment_result.outputs.dartel_input_images[
        1]

    results = joblib.Parallel(
        n_jobs=N_JOBS, verbose=100,
        pre_dispatch='1.5*n_jobs',  # for scalability over RAM
        )(joblib.delayed(
        _do_subject_dartelnorm2mni)(
              subject_output_dirs[j],
              structural_files[j],
              functional_files[j],
              subject_gm_file=subject_gm_files[j],
              subject_wm_file=subject_wm_files[j],
              sessions=session_ids[j],
              subject_id=subject_ids[j],
              do_report=do_report,
              do_cv_tc=do_cv_tc,
              final_thumbnail=subject_final_thumbs[j],
              results_gallery=subject_results_galleries[j],
              subject_progress_logger=subject_progress_loggers[j],
              parent_results_gallery=parent_results_gallery,
              ignore_exception=ignore_exception,
              modulate=False,  # don't modulate
              fwhm=fwhm,
              flowfield_files=dartel_result.outputs.dartel_flow_fields[j],
              template_file=dartel_result.outputs.final_template_file,
              )
          for j in xrange(
              len(subject_ids)))

    # do QA
    if do_report:
        pass

    return results, newsegment_result


def do_subjects_preproc(subjects,
                        output_dir=None,
                        subject_callback=None,
                        dataset_id="UNNAMED DATASET!",
                        do_deleteorient=False,
                        do_report=True,
                        do_export_report=False,
                        dataset_description=None,
                        report_filename=None,
                        do_shutdown_reloaders=True,
                        fwhm=0,
                        do_bet=False,
                        do_slicetiming=False,
                        do_realign=True,
                        do_coreg=True,
                        do_segment=True,
                        do_normalize=True,
                        do_dartel=False,
                        do_cv_tc=True,
                        ignore_exception=True,
                        n_jobs=N_JOBS,
                        ):

    """This functions doe intra-subject fMRI preprocessing on a
    group os subjects.

    Parameters
    ----------
    subjects: iterable of SubjectData objects

    report_filename: string (optional)
    if provided, an HTML report will be produced. This report is
    dynamic and its contents are updated automatically as more
    and more subjects are preprocessed.

    Returns
    -------
    list of Bunch dicts with keys: anat, func, and subject_id, etc.
    for each preprocessed subject

    """

    # sanitize input
    subjects = list(subjects)

    if do_dartel:
        do_segment = False
        do_normalize = False

    # if do_report and report_filename is None:
    #     raise RuntimeError(
    #         ("You asked for reporting (do_report=True)  but specified"
    #          " an invalid report_filename (None)"))

    kwargs = {'do_deleteorient': do_deleteorient,
              'do_bet': do_bet,
              'do_report': do_report,
              'do_slicetiming': do_slicetiming,
              'do_realign': do_realign, 'do_coreg': do_coreg,
              'do_segment': do_segment, 'do_normalize': do_normalize,
              'do_cv_tc': do_cv_tc,
              'fwhm': fwhm,
              'ignore_exception': ignore_exception,
              'last_stage': not do_dartel,
              }

    if output_dir is None:
        output_dir = os.path.abspath("/tmp/runs_XYZ")
        print "Output directory set to: %s" % output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    additional_preproc_undergone = ""
    if do_dartel:
        additional_preproc_undergone += (
            "<li>"
            "Group/Inter-subject Normalization has been done using the "
            "SPM8 <a href='%s'>DARTEL</a> to warp subject brains into "
            "MNI space. "
            "The idea is to register images by computing a &ldquo;flow"
            " field&rdquo; which can then be &ldquo;exponentiated"
            "&rdquo; to generate both forward and backward deformation"
            "s. Processing begins with the &ldquo;import&rdquo; "
            "step. This involves taking the parameter files "
            "produced by the segmentation (NewSegment), and writing "
            "out rigidly "
            "transformed versions of the tissue class images, "
            "such that they are in as close alignment as possible with"
            " the tissue probability maps. &nbsp; "
            "The next step is the registration itself. This involves "
            "the simultaneous registration of e.g. GM with GM, "
            "WM with WM and 1-(GM+WM) with 1-(GM+WM) (when needed, the"
            " 1- (GM+WM) class is generated implicitly, so there "
            "is no need to include this class yourself). This "
            "procedure begins by creating a mean of all the images, "
            "which is used as an initial template. Deformations "
            "from this template to each of the individual images "
            "are computed, and the template is then re-generated"
            " by applying the inverses of the deformations to "
            "the images and averaging. This procedure is repeated a "
            "number of times. &nbsp;Finally, warped "
            "versions of the images (or other images that are in "
            "alignment with them) can be generated. "
            "</li>") % DARTEL_URL

    # get caller module handle from stack-frame
    user_script_name = sys.argv[0]
    user_source_code = base_reporter.get_module_source_code(
        user_script_name)

    preproc_undergone = """\
    <p>All preprocessing has been done using the <i>%s</i> script of
 <a href="%s">pypreprocess</a>, which is powered by
 <a href="%s">nipype</a>, and <a href="%s">SPM8</a>.
    </p>""" % (user_script_name, PYPREPROCESS_URL,
               NIPYPE_URL, SPM8_URL)

    preproc_undergone += "<ul>"

    preproc_undergone += additional_preproc_undergone + "</ul>"

    kwargs['additional_preproc_undergone'] = additional_preproc_undergone

    # generate html report (for QA) as desired
    parent_results_gallery = None
    if do_report:
        # copy css and js stuff to output dir
        for js_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                          "js/*.js")):
            shutil.copy(js_file, output_dir)
        for css_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                          "css/*.css")):
            shutil.copy(css_file, output_dir)
        for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                                "icons/*.gif")):
            shutil.copy(icon_file, output_dir)
        for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                                "images/*.png")):
            shutil.copy(icon_file, output_dir)
        for icon_file in glob.glob(os.path.join(base_reporter.ROOT_DIR,
                                                "images/*.jpeg")):
            shutil.copy(icon_file, output_dir)

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
                           ) % (preproc_func_name,
                                user_script_name)
        preproc_params += base_reporter.dict_to_html_ul(
            dict((arg, values[arg]) for arg in args if not arg in [
                    "dataset_description",
                    "report_filename",
                    "do_report",
                    "do_cv_tc",
                    "do_export_report",
                    "do_shutdown_reloaders",
                    "subjects",
                    # add other args to exclude below
                    ]
                 ))

        # initialize results gallery
        loader_filename = os.path.join(
            output_dir, "results_loader.php")
        parent_results_gallery = base_reporter.ResultsGallery(
            loader_filename=loader_filename,
            refresh_timeout=30,
            )

        # initialize progress bar
        progress_logger = base_reporter.ProgressReport(
            report_log_filename,
            other_watched_files=[report_filename,
                                 report_preproc_filename])

        # html markup
        log = base_reporter.get_dataset_report_log_html_template(
            ).substitute(
            start_time=time.ctime(),
            )

        preproc = base_reporter.get_dataset_report_preproc_html_template(
            ).substitute(
            results=parent_results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            dataset_description=dataset_description,
            source_code=user_source_code,
            source_script_name=user_script_name,
            preproc_params=preproc_params,
            )

        main_html = base_reporter.get_dataset_report_html_template(
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

            if do_shutdown_reloaders:
                progress_logger.finish_all()

        kwargs['parent_results_gallery'] = parent_results_gallery

    results = joblib.Parallel(
        n_jobs=n_jobs,
        pre_dispatch='1.5*n_jobs',  # for scalability over RAM
        verbose=100)(joblib.delayed(
            _do_subject_preproc)(
                subject_data, **kwargs) for subject_data in subjects)

    subject_ids = [output['subject_id'] for _, output in results]

    # collect subject output dirs
    subject_output_dirs = [output['output_dir'] for _, output in results]

    if do_dartel:
        # collect subject_ids and session_ids
        session_ids = [output['session_id'] for _, output in results]
        _preproc_undergone = dict((output['subject_id'],
                                   output['preproc_undergone'])
                                  for _, output in results)
        # collect estimated motion
        if do_realign:
            estimated_motion = dict((output["subject_id"],
                                     output['estimated_motion'])
                                    for _, output in results)

        # collect structural files for DARTEL pipeline
        if do_coreg:
            structural_files = [
                output['coreg_result'].outputs.coregistered_source
                for _, output in results]
        else:
            structural_files = [
                subject_data.anat for subject_data, _ in results]

        # collect functional files for DARTEL pipeline
        if do_realign:
            functional_files = [
                output['realign_result'].outputs.realigned_files
                for _, output in results]
        else:
            functional_files = [output['func'] for _, output in results]

        # collect gallery related subject-specific stuff
        subject_final_thumbs = None
        subject_results_galleries = None
        subject_progress_loggers = None

        if do_report:
            subject_final_thumbs = [output['final_thumbnail']
                                    for _, output in results]
            subject_results_galleries = [output['results_gallery']
                                         for _, output in results]
            subject_progress_loggers = [output['progress_logger']
                                        for _, output in results]
            if do_realign:
                estimated_motion = dict((output["subject_id"],
                                         output['estimated_motion'])
                                        for _, output in results)

        # normalize brains to their own template space (DARTEL)
        results, newsegment_result = do_group_DARTEL(
            output_dir,
            subject_ids,
            session_ids,
            structural_files,
            functional_files,
            fwhm=fwhm,
            subject_output_dirs=subject_output_dirs,
            do_report=do_report,
            do_cv_tc=do_cv_tc,
            subject_final_thumbs=subject_final_thumbs,
            subject_results_galleries=subject_results_galleries,
            subject_progress_loggers=subject_progress_loggers,
            parent_results_gallery=parent_results_gallery,
            ignore_exception=ignore_exception,
            )

        # collect results
        _results = []

        if results:
            for item in results:
                if item:
                    subject_result = {}
                    subject_result['preproc_undergone'] = _preproc_undergone[
                        item['subject_id']]
                    subject_result['subject_id'] = item['subject_id']

                    # output final fmri and anat image filenames
                    subject_result['func'] = item['func']
                    subject_result['anat'] = item['anat']

                    # output segmented GM compartment
                    subject_result['gm'] = item['gm']

                    # output segmented WM compartment
                    subject_result['wm'] = item['wm']

                    # output motion parameters
                    if do_realign:
                        subject_result['estimated_motion'] = estimated_motion[
                            item['subject_id']]
                    subject_result['output_dir'] = item['output_dir']

                    json_output_filename = os.path.join(
                        subject_result['output_dir'], 'infos_DARTEL.json')
                    json.dump(subject_result, open(json_output_filename, 'wb'))

                    if do_report:
                        subject_result['progress_logger'] = item[
                            'progress_logger']

                        if do_shutdown_reloaders:
                            subject_result['progress_logger'].finish_all()

                    _results.append(subject_result)

        results = _results
    else:
        # collect results
        _results = []

        if results:
            for x in results:
                if x is None:
                    continue
                else:
                    item = x[1]
                subject_result = {}
                subject_result['preproc_undergone'] = item['preproc_undergone']
                subject_result['subject_id'] = item['subject_id']
                subject_result['func'] = item['func']
                if 'anat' in item.keys():
                    subject_result['anat'] = item['anat']
                if do_realign:
                    subject_result['estimated_motion'] = item[
                        'estimated_motion']
                if do_segment:
                    # output segmented GM compartment
                    subject_result['gm'] = item['gm']

                    # output segmented WM compartment
                    subject_result['wm'] = item['wm']

                subject_result['output_dir'] = item['output_dir']

                json_output_filename = os.path.join(
                    subject_result['output_dir'], 'infos.json')
                json.dump(subject_result, open(json_output_filename, 'wb'))

                if do_report:
                    subject_result['progress_logger'] = item['progress_logger']

                    if do_shutdown_reloaders:
                        subject_result['progress_logger'].finish_all()

                _results.append(subject_result)

        results = _results

    if do_report:
        finalize_report()

        print "HTML report (dynamic) written to %s" % report_filename

    # export report (so it can be emailed and commented offline, for example)
    if do_report and do_export_report:
        tag = "DARTEL" if do_dartel else "standard"

        # dst dir for the frozen reports
        frozen_report_dir = os.path.join(output_dir,
                                         "frozen_report_%s" % tag)

        # copy group-level report files
        base_reporter.copy_report_files(output_dir, frozen_report_dir)

        # copy subject-level report files
        for subject_output_dir in subject_output_dirs:
            dst = os.path.join(frozen_report_dir,
                                  os.path.basename(subject_output_dir))
            base_reporter.copy_report_files(subject_output_dir, dst)

    # return preproc results
    return results
