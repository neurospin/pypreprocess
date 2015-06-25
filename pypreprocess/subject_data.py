"""
Encapsulation of subject data. Handles subject data logic
like filetypes, extensions, .nii.gz -> .nii conversions, setup and
teardown for reports, etc., general sanitization, etc. Also handles
progressive report generation for subject during preprocessing.

"""
# Author: DOHMATOB Elvis Dopgima

import os
import time
import warnings
import numpy as np
from matplotlib.pyplot import cm
from pypreprocess.external.joblib import Memory
from .io_utils import (niigz2nii as do_niigz2nii, dcm2nii as do_dcm2nii,
                       isdicom, delete_orientation, hard_link, get_shape,
                       is_4D, is_3D, get_basenames, is_niimg)
from .reporting.base_reporter import (
    commit_subject_thumnbail_to_parent_gallery,
    ResultsGallery, Thumbnail, a, img, copy_web_conf_files,
    ProgressReport, get_subject_report_html_template,
    get_subject_report_preproc_html_template, copy_failed_png)

from .reporting.preproc_reporter import (generate_tsdiffana_thumbnail,
                                         generate_realignment_thumbnails,
                                         generate_coregistration_thumbnails,
                                         generate_normalization_thumbnails,
                                         generate_segmentation_thumbnails,
                                         make_nipype_execution_log_html)


# tooltips for thumbnails in report pages
mc_tooltip = ("Motion parameters estimated during motion-"
              "correction. If motion is less than half a "
              "voxel, it's generally OK. Moreover, it's "
              "recommended to include these estimated motion "
              "parameters as confounds (nuissance regressors) "
              "in the the GLM.")
segment_acronyms = ("Acronyms: TPM means Tissue Probability Map; GM means "
                   "Grey-Matter;"
                   " WM means White-Matter; CSF means Cerebro-Spinal Fuild")
reg_tooltip = ("The red contours should"
               " match the background image well. Otherwise, something might"
               " have gone wrong. Typically things that can go wrong include: "
               "lesions (missing brain tissue); bad orientation headers; "
               "non-brain tissue in anatomical image, etc. In rare cases, it "
               "might be that the registration algorithm simply didn't "
               "succeed.")
segment_tooltip = ("%s. The TPM contours shoud match the background image "
                   "well. Otherwise, something might have gone wrong. "
                   "Typically things that can go wrong include: "
                   "lesions (missing brain tissue); bad orientation headers; "
                   "non-brain tissue in anatomical image (i.e needs brain "
                   "extraction), etc. In rare cases, it might be that the"
                   " segmentation algorithm simply didn't succeed." % (
                       segment_acronyms))
tsdiffana_tooltips = [
    ("(Squared) differences across sequential volumes. "
     "A large value indicates an artifact that occurred during the "
     "slice acquisition, possibly related to motion."),
    ("Average signal over each volume. A large drop/peak (e.g. 5%) indicates "
     "an artefact."),
    ("Variance index per slice. Note that aqquisition artifacts can be slice"
     "-specific. Look at the data if there is a peak somewhere."),
    ("Scaled variance per slice indicates slices where artifacts occur."
    "A slice/time with large variance should be eyeballed."),
    ("Large variations should be confined to vascular structures "
     "or ventricles. Large variations around the brain indicate"
     " (uncorrected) motion effects."),
    ("Large variations should be confined to vascular structures or"
     " ventricles. Large variations around the brain indicate (uncorrected)"
     " motion effects.")]


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

    session_ids: string or list of strings, optional (default None):
        session ids for all sessions (i.e runs)

    output_dir: string, optional (default None)
        output directory for this subject; will be create if doesn't exist

    session_output_dirs: list of strings, optional (default None):
        list of output directories, one for each session

    anat_output_dir: string, optional (default None)
        output directory of anatomical data

    scratch: string, optional (default None)
        root directory for scratch data (temp files, cache, etc.) for this
        subject; thus we can push all scratch data unto a dedicated device

    **kwargs: param-value dict_like
        additional optional parameters

    """

    def __init__(self, func=None, anat=None, subject_id="sub001",
                 session_ids=None, output_dir=None, session_output_dirs=None,
                 anat_output_dir=None, scratch=None, warpable=None, **kwargs):
        if warpable is None:
            warpable = ['anat', 'func']
        self.func = func
        self.anat = anat
        self.subject_id = subject_id
        self.session_ids = session_ids
        self.n_sessions = None
        self.output_dir = output_dir
        self.anat_output_dir = anat_output_dir
        self.session_output_dirs = session_output_dirs
        self.scratch = scratch
        self.tmp_output_dir = None
        self.warpable = warpable
        self.failed = False
        self.warpable = warpable
        self.nipype_results = {}
        self._set_items(**kwargs)

    def _set_items(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _delete_orientation(self):
        """
        Delete orientation metadata. Garbage orientation metadata can lead to
        severe mis-registration trouble.

        """

        # prepare for smart caching
        cache_dir = os.path.join(self.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        mem = Memory(cachedir=cache_dir, verbose=5)

        # deleteorient for func
        for attr in ['n_sessions', 'session_output_dirs']:
            if getattr(self, attr) is None:
                warnings.warn("'%s' attribute of is None! Skipping" % attr)
                break
        else:
            self.func = [mem.cache(delete_orientation)(
                self.func[sess], self.session_output_dirs[sess])
                         for sess in range(self.n_sessions)]

        # deleteorient for anat
        if not self.anat is None:
            self.anat = mem.cache(delete_orientation)(
                self.anat, self.anat_output_dir)

    def _sanitize_output_dir(self, output_dir):
        if not output_dir is None:
            output_dir = os.path.abspath(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        return output_dir

    def _sanitize_session_output_dirs(self):
        """func output dirs (one per session)"""
        if self.session_output_dirs is None:
            if self.n_sessions is None:
                return
            self.session_output_dirs = [None] * self.n_sessions

        # session-wise func output directories
        for sess, sess_output_dir in enumerate(self.session_output_dirs):
            if sess_output_dir is None:
                if self.n_sessions > 1:
                    sess_output_dir = os.path.join(
                        self.output_dir, self.session_ids[sess])
                else:
                    sess_output_dir = self.output_dir
            else:
                sess_output_dir = sess_output_dir
            self.session_output_dirs[sess] = self._sanitize_output_dir(
                sess_output_dir)

    def _sanitize_output_dirs(self):
        # output dir
        self.output_dir = self._sanitize_output_dir(self.output_dir)

        # anat output dir
        if self.anat_output_dir is None:
            self.anat_output_dir = self.output_dir
        self.anat_output_dir = self._sanitize_output_dir(self.anat_output_dir)

        # sanitize per-session func output dirs
        self._sanitize_session_output_dirs()

        # make tmp output dir
        if self.scratch is None:
            self.scratch = self.output_dir
        if not self.scratch is None:
            self.tmp_output_dir = os.path.join(self.scratch, "tmp")
        self.tmp_output_dir = self._sanitize_output_dir(self.tmp_output_dir)

    def _niigz2nii(self):
        """
        Convert .nii.gz to .nii (crucial for SPM).

        """
        cache_dir = os.path.join(self.scratch, 'cache_dir')
        mem = Memory(cache_dir, verbose=100)
        self._sanitize_session_output_dirs()
        if not None in [self.func, self.n_sessions, self.session_output_dirs]:
            self.func = [mem.cache(do_niigz2nii)(
                self.func[sess], output_dir=self.session_output_dirs[sess])
                         for sess in range(self.n_sessions)]
        if not self.anat is None:
            self.anat = mem.cache(do_niigz2nii)(
                self.anat, output_dir=self.anat_output_dir)

    def _dcm2nii(self):
        """
        Convert DICOM to nifti.

        """
        self.isdicom = False
        if self.func:
            if not isinstance(self.func[0], basestring):
                if not is_niimg(self.func[0]):
                    self.isdicom = isdicom(self.func[0][0])
            self.func = [do_dcm2nii(sess_func, output_dir=self.output_dir)[0]
                         for sess_func in self.func]
        if self.anat:
            self.anat = do_dcm2nii(self.anat, output_dir=self.output_dir)[0]

    def _check_func_names_and_shapes(self):
        """
        Checks that abspaths of func imagesare distinct with and across
        sessions, and each session should constitute a 4D film (as a
        string or list of)

        """
        # check that functional image abspaths are distinct across sessions
        for sess1 in range(self.n_sessions):
            if is_niimg(self.func[sess1]):
                continue

            # functional images for this session must all be distinct abspaths
            if not isinstance(self.func[sess1], basestring):
                if len(self.func[sess1]) != len(set(self.func[sess1])):
                    # Oops! there must be a repetition somewhere
                    for x in self.func[sess1]:
                        count = self.func[sess1].count(x)
                        if count > 1:
                            rep = x
                            break
                    raise RuntimeError(
                        "List of functional images for session number %i"
                        " has the file %s repeated %s times" % (
                            sess1 + 1, rep, count))

            # all functional data for this session should constitute a 4D film
            if isinstance(self.func[sess1], basestring):
                if not is_4D(self.func[sess1]):
                    raise RuntimeError(
                        "Functional images for session number %i"
                        " doesn't constitute a 4D film; the shape of the"
                        " session is %s; the images for this session are "
                        "%s" % (sess1 + 1, get_shape(self.func[sess1]),
                                self.func[sess1]))
            else:
                for x in self.func[sess1]:
                    if not is_3D(x):
                        raise RuntimeError(
                            "Image %s of session number %i is not 3D; it "
                            "has shape %s." % (x, sess1 + 1, get_shape(x)))

            # functional images for sess1 shouldn't concide with any functional
            # image of any other session
            for sess2 in range(sess1 + 1, self.n_sessions):
                if is_niimg(self.func[sess2]):
                    continue
                if self.func[sess1] == self.func[sess2]:
                    raise RuntimeError(
                        ('The same image %s specified for session number %i '
                         'and %i' % (self.func[sess1], sess1 + 1,
                                     sess2 + 1)))
                if isinstance(self.func[sess1], basestring):
                    if self.func[sess1] == self.func[sess2]:
                        raise RuntimeError(
                            'The same image %s specified for session '
                            "number %i and %i" % (
                                self.func[sess1], sess1 + 1, sess2 + 1))
                else:
                    if not isinstance(self.func[sess2], basestring):
                        if self.func[sess2] in self.func[sess1]:
                            raise RuntimeError(
                                'The same image %s specified for session'
                                ' number %i and %i' % (
                                    self.func[sess1], sess1 + 1, sess2 + 1))
                        else:
                            for x in self.func[sess1]:
                                if is_niimg(x):
                                    continue
                                for y in self.func[sess2]:
                                    if is_niimg(y):
                                        continue
                                    if x == y:
                                        raise RuntimeError(
                                            'The same image %s specified for '
                                            'in both session number %i '
                                            'and %i' % (x, sess1 + 1,
                                                        sess2 + 1))

    def _set_session_ids(self):
        if self.func is None:
            return
        elif isinstance(self.func, basestring): self.func = [self.func]
        if self.session_ids is None:
            if len(self.func) > 10:
                raise RuntimeError
            self.session_ids = ["Session%i" % (sess + 1)
                                for sess in range(len(self.func))]
        else:
            if isinstance(self.session_ids, (basestring, int)):
                assert len(self.func) == 1
                self.session_ids = [self.session_ids]
            else:
                assert len(self.session_ids) == len(self.func), "%s != %s" % (
                    len(self.session_ids), len(self.func))
        self.n_sessions = len(self.session_ids)

    def sanitize(self, deleteorient=False, niigz2nii=False):
        """
        This method does basic sanitization of the `SubjectData` instance, like
        extracting .nii.gz -> .nii (crusial for SPM), ensuring that functional
        images actually exist on disk, etc.

        Parameters
        ----------
        deleteorient: bool (optional)
            if true, then orientation meta-data in all input image files
            for this subject will be stripped-off

        niigz2nii: bool, optional (default False)
            convert func and ant .nii.gz images to .nii

        """
        # sanitize output_dir
        self._sanitize_output_dirs()

        # sanitize func
        if isinstance(self.func, basestring):
            self.func = [self.func]

        # sanitize anat
        if not self.anat is None:
            if not os.path.isfile(self.anat):
                raise OSError("%s is not a file!" % self.anat)

        # sanitize session_ids
        if self.func is None:
            return self
        self._set_session_ids()

        # .dcm, .ima -> .nii
        self._dcm2nii()

        # delete orientation meta-data
        if deleteorient:
            self._delete_orientation()

        # .nii.gz -> .nii extraction for SPM & co.
        if niigz2nii:
            self._niigz2nii()

        self._check_func_names_and_shapes()

        # get basenames
        self.basenames = [get_basenames(self.func[sess]) if not is_niimg(
                self.func[sess]) else "%s.nii.gz" % self.session_ids[sess]
                        for sess in range(self.n_sessions)]

        self._set_session_ids()
        return self

    def save_realignment_parameters(self, lkp=6):
        if not hasattr(self, "realignment_parameters"):
            return
        rp_filenames = []
        for sess in range(self.n_sessions):
            sess_rps = getattr(self, "realignment_parameters")[sess]
            if isinstance(sess_rps, basestring):
                rp_filenames.append(sess_rps)
            else:
                sess_basename = self.basenames[sess]
                if not isinstance(sess_basename, basestring):
                    sess_basename = sess_basename[0]

                rp_filename = os.path.join(
                    self.tmp_output_dir,
                    "rp_" + sess_basename + ".txt")
                np.savetxt(rp_filename, sess_rps[..., :lkp])
                rp_filenames.append(rp_filename)
        setattr(self, "realignment_parameters", rp_filenames)
        return rp_filenames

    def hardlink_output_files(self, final=False):
        """
        Hard-links output files to subject's immediate output directory.

        Parameters
        ----------
        final: bool, optional (default False)
            flag indicating whether, we're finalizing the preprocessing
            pipeline

        """
        self._set_session_ids()

        # anat stuff
        for item in ["anat", 'gm', 'wm', 'csf', 'wgm', 'wwm', 'wcsf',
                     'mwgm', 'mwwm', 'mwcsf']:
            if hasattr(self, item):
                filename = getattr(self, item)
                if not filename is None:
                    linked_filename = hard_link(filename, self.anat_output_dir)
                    if final: setattr(self, item, linked_filename)

        # func stuff
        self.save_realignment_parameters()
        for item in ['func', 'realignment_parameters']:
            tmp = []
            if hasattr(self, item):
                filenames = getattr(self, item)
                if isinstance(filenames, basestring):
                    assert self.n_sessions == 1, filenames
                    filenames = [filenames]
                for sess in range(self.n_sessions):
                    filename = filenames[sess]
                    if not filename is None:
                        linked_filename = hard_link(
                            filename, self.session_output_dirs[sess])
                        tmp.append(linked_filename)
            if final:
                setattr(self, item, tmp)

    def init_report(self, parent_results_gallery=None,
                    tsdiffana=True, preproc_undergone=None):
        """
        This method is invoked to initialize the reports factory for the
        subject. It configures everything necessary for latter reporting:
        copies the custom .css, .js, etc. files, generates markup for
        the report pages (report.html, report_log.html, report_preproc.html,
        etc.), etc.

        Parameters
        ----------
        parent_results_gallery: reporting.base_reporter.ResultsGallery instance

        tsdiffana: bool, optional
            if set, six figures are added to characterize differences between
            consecutive time points in the times series for artefact detection
        preproc_undergone: list of srtings,
            list of processing steps performed

        """
        # misc
        self._set_session_ids()
        if not self.func:
            tsdiffana = False

        # make sure output_dir is OK
        self._sanitize_output_dirs()

        # misc for reporting
        self.reports_output_dir = os.path.join(self.output_dir, "reports")
        if not os.path.exists(self.reports_output_dir):
            os.makedirs(self.reports_output_dir)
        self.report = True
        self.results_gallery = None
        self.parent_results_gallery = parent_results_gallery
        self.tsdiffana = tsdiffana

        # report filenames
        self.report_log_filename = os.path.join(
            self.reports_output_dir, 'report_log.html')
        self.report_preproc_filename = os.path.join(
            self.reports_output_dir, 'report_preproc.html')
        self.report_filename = os.path.join(self.reports_output_dir,
                                            'report.html')

        # clean report files
        open(self.report_log_filename, 'w').close()
        open(self.report_preproc_filename, 'w').close()
        open(self.report_filename, 'w').close()

        # initialize results gallery
        loader_filename = os.path.join(self.reports_output_dir,
                                       "results_loader.php")
        self.results_gallery = ResultsGallery(
            loader_filename=loader_filename,
            title="Report for subject %s" % self.subject_id)

        # final thumbnail most representative of this subject's QA
        self.final_thumbnail = Thumbnail()
        self.final_thumbnail.a = a(
            href=self.report_preproc_filename)
        self.final_thumbnail.img = img(src=None)
        self.final_thumbnail.description = self.subject_id

        # copy web stuff to subject output dir
        copy_web_conf_files(self.reports_output_dir)

        # initialize progress bar
        self.progress_logger = ProgressReport(
            self.report_log_filename,
            other_watched_files=[self.report_preproc_filename])

        # html markup
        preproc = get_subject_report_preproc_html_template(
            results=self.results_gallery, start_time=time.ctime(),
            preproc_undergone=preproc_undergone, subject_id=self.subject_id)
        main_html = get_subject_report_html_template(
            start_time=time.ctime(), subject_id=self.subject_id)
        with open(self.report_preproc_filename, 'w') as fd:
            fd.write(str(preproc))
            fd.close()
        with open(self.report_filename, 'w') as fd:
            fd.write(str(main_html))
            fd.close()

    def finalize_report(self, parent_results_gallery=None, last_stage=False):
        """
        Finalizes the business of reporting.

        """
        # misc
        self._set_session_ids()
        if not self.reporting_enabled():
            return
        if parent_results_gallery is None:
            parent_results_gallery = self.parent_results_gallery

        # generate failure thumbnail
        if self.failed:
            copy_failed_png(self.reports_output_dir)
            self.final_thumbnail.img.src = 'failed.png'
            self.final_thumbnail.description += (
                ' (failed preprocessing)')
        else:
            # generate tsdiffana plots
            if self.tsdiffana:
                generate_tsdiffana_thumbnail(
                    self.func, self.session_ids, self.subject_id,
                    self.reports_output_dir, tooltips=tsdiffana_tooltips,
                    results_gallery=self.results_gallery)

        # shut down all watched report pages
        self.progress_logger.finish_all()

        # shut down everything in reports_output_dir
        if last_stage:
            self.progress_logger.finish_dir(self.reports_output_dir)

        # commit final thumbnail into parent's results gallery
        if not parent_results_gallery is None:
            commit_subject_thumnbail_to_parent_gallery(
                self.final_thumbnail, self.subject_id, parent_results_gallery)

        print ("\r\nPreproc report for subject %s written to %s"
               " .\r\n" % (self.subject_id,
                           self.report_preproc_filename))

    def __getitem__(self, key):
        return self.__dict__[key]

    def reporting_enabled(self):
        """
        Determines whether reporting is enabled for this subject

        """
        return hasattr(self, 'results_gallery')

    def generate_realignment_thumbnails(self, log=True, nipype=True):
        """
        Invoked to generate post-realignment thumbnails.

        """
        # misc
        self._set_session_ids()
        if not hasattr(self, 'realignment_parameters'):
            raise ValueError("'realignment_parameters' attribute not set!")
        if not self.reporting_enabled():
            self.init_report()

        # log execution
        if log:
            execution_log_html = None
            if nipype:
                execution_log_html = make_nipype_execution_log_html(
                    self.func, "Motion_Correction",
                    self.reports_output_dir)
            self.progress_logger.log(
                "<b>Motion Correction</b><br/>")
            if execution_log_html:
                self.progress_logger.log(open(execution_log_html).read())
                self.progress_logger.log('<hr/>')

        # generate thumbnails proper
        thumbs = generate_realignment_thumbnails(
            getattr(self, 'realignment_parameters'),
            self.reports_output_dir, sessions=self.session_ids,
            execution_log_html_filename=execution_log_html if log
            else None, tooltip=mc_tooltip,
            results_gallery=self.results_gallery)
        self.final_thumbnail.img.src = thumbs['rp_plot']

    def generate_coregistration_thumbnails(self, coreg_func_to_anat=True,
                                           log=True, comment=True,
                                           nipype=True):
        """
        Invoked to generate post-coregistration thumbnails.

        """
        # misc
        self._set_session_ids()
        if self.anat is None:
            print("Subject 'anat' field is None; nothing to do")
            return

        # misc
        if not self.reporting_enabled():
            self.init_report()
        src, ref = self.func, self.anat
        src_brain, ref_brain = "mean functional image", "anatomical image"
        if not coreg_func_to_anat:
            src, ref = ref, src
            src_brain, ref_brain = ref_brain, src_brain

        # log execution
        if log:
            execution_log_html = None
            if nipype:
                execution_log_html = make_nipype_execution_log_html(
                    src, "Coregister",
                    self.reports_output_dir)
            self.progress_logger.log(
                "<b>Coregistration</b><br/>")
            if execution_log_html:
                self.progress_logger.log(open(execution_log_html).read())
                self.progress_logger.log('<hr/>')

        # generate thumbs proper
        thumbs = generate_coregistration_thumbnails(
            (ref, ref_brain), (src, src_brain),
            self.reports_output_dir,
            execution_log_html_filename=execution_log_html if log
            else None, results_gallery=self.results_gallery,
            comment=comment, tooltip=reg_tooltip)
        self.final_thumbnail.img.src = thumbs['axial']

    def generate_segmentation_thumbnails(self, log=True, nipype=True):
        """
        Invoked to generate post-segmentation thumbnails.

        """
        # misc
        self._set_session_ids()
        segmented = False
        for item in ['gm', 'wm', 'csf']:
            if hasattr(self, item):
                segmented = True
                break
        if not segmented:
            return
        if not self.reporting_enabled():
            self.init_report()

        # log execution
        if log:
            execution_log_html = None
            if nipype:
                execution_log_html = make_nipype_execution_log_html(
                    getattr(self, 'gm') or getattr(self, 'wm') or getattr(
                        self, 'csf'), "Segment", self.reports_output_dir)
            self.progress_logger.log(
                "<b>Segmentation</b><br/>")
            if execution_log_html:
                self.progress_logger.log(open(execution_log_html).read())
                self.progress_logger.log('<hr/>')

        # generate thumbnails proper
        for brain_name, brain, cmap in zip(
                ['anatomical image', 'mean functional image'],
                [self.anat, self.func], [cm.gray, cm.spectral]):
            if not brain: continue
            thumbs = generate_segmentation_thumbnails(
                brain, self.reports_output_dir,
                subject_gm_file=getattr(self, 'gm', None),
                subject_wm_file=getattr(self, 'wm', None),
                subject_csf_file=getattr(self, 'csf', None),
                cmap=cmap, brain=brain_name, only_native=True,
                execution_log_html_filename=execution_log_html if log
                else None, results_gallery=self.results_gallery,
                tooltip=segment_tooltip)
            if 'func' in brain_name:
                self.final_thumbnail.img.src = thumbs['axial']

    def generate_normalization_thumbnails(self, log=True, nipype=True):
        """
        Invoked to generate post-normalization thumbnails.

        """
        # misc
        self._set_session_ids()
        if not self.reporting_enabled():
            self.init_report()
        warped_tpms = dict(
            (tpm, getattr(self, tpm, getattr(self, "m" + tpm, None)))
            for tpm in ["gm", "wm", "csf"])
        segmented = warped_tpms.values().count(None) < len(warped_tpms)

        # generate thumbnails proper
        for brain_name, brain, cmap in zip(
                ['anatomical image', 'mean functional image'],
                [self.anat, self.func], [cm.gray, cm.spectral]):
            if not brain:
                continue

            # generate segmentation thumbs
            if segmented:
                thumbs = generate_segmentation_thumbnails(
                    brain, self.reports_output_dir,
                    subject_gm_file=warped_tpms["gm"],
                    subject_wm_file=warped_tpms["wm"],
                    subject_csf_file=warped_tpms["csf"],
                    cmap=cmap, brain=brain_name, comments="warped",
                    execution_log_html_filename=make_nipype_execution_log_html(
                        warped_tpms["gm"] or warped_tpms["wm"] or
                        warped_tpms["csf"],
                        "Segment", self.reports_output_dir) if log else None,
                    results_gallery=self.results_gallery,
                    tooltip=segment_tooltip)
                if 'func' in brain_name or not self.func:
                    self.final_thumbnail.img.src = thumbs['axial']

            # log execution
            if log:
                execution_log_html = None
                if nipype:
                    execution_log_html = make_nipype_execution_log_html(
                        brain, "Normalization", self.reports_output_dir,
                        brain_name)
                self.progress_logger.log(
                    "<b>Normalization of %s</b><br/>" % brain_name)
                if execution_log_html:
                    text = open(execution_log_html).read()
                    if "normalized_files" in text or "warped_files" in text:
                        self.progress_logger.log(text)
                        self.progress_logger.log('<hr/>')

            # generate normalization thumbs proper
            thumbs = generate_normalization_thumbnails(
                brain, self.reports_output_dir, brain=brain_name,
                execution_log_html_filename=execution_log_html if log
                else None, results_gallery=self.results_gallery,
                tooltip=reg_tooltip)
            if not segmented and ("func" in brain_name or not self.func):
                self.final_thumbnail.img.src = thumbs['axial']

    def generate_smooth_thumbnails(self):
        """
        Generate thumbnails post-smoothing.
        """
        # misc
        if self.func is None: return
        if not self.reporting_enabled():
            self.init_report()

        # log execution
        title = "func smooth"
        execution_log_html = make_nipype_execution_log_html(
            self.func, title, self.reports_output_dir)
        self.progress_logger.log(
            "%s</b><br/>" % title)
        text = open(execution_log_html).read()
        if "smoothed_files" in text:
            self.progress_logger.log(text)
        self.progress_logger.log('<hr/>')

    def generate_report(self):
        """
        Method invoked to generate all reports in one-go. This is useful
        for generating reports out-side the preproc logic: simply populate
        the func, anat (optionally realignment_params, gm, wm, csf, etc.)
        fields and then fire this method.

        """
        # misc
        self._set_items()
        self.sanitize()

        # report proper
        self.generate_realignment_thumbnails(log=False)
        self.generate_coregistration_thumbnails(log=False, comment=False)
        self.generate_normalization_thumbnails(log=False)
        self.finalize_report(last_stage=True)

    def __repr__(self):
        return repr(self.__dict__)
