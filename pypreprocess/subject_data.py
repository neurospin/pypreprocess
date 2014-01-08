"""
:Synopsis: Encapsulation of subject data. Handles subject data logic
like filetypes, extensions, .nii.gz -> .nii conversions, setup and
teardown for reports, etc., general sanitization, etc.
:Author: DOHMATOB Elvis Dopgima

"""

import os
import time
from joblib import Memory
from matplotlib.pyplot import cm
from .io_utils import (niigz2nii as do_niigz2nii,
                       dcm2nii as do_dcm2nii,
                       isdicom,
                       delete_orientation,
                       hard_link)
from .reporting.base_reporter import (
    commit_subject_thumnbail_to_parent_gallery,
    ResultsGallery,
    Thumbnail,
    a,
    img,
    copy_web_conf_files,
    ProgressReport,
    get_subject_report_html_template,
    get_subject_report_preproc_html_template,
    copy_failed_png
    )
from .reporting.preproc_reporter import (
    generate_cv_tc_thumbnail,
    generate_realignment_thumbnails,
    generate_coregistration_thumbnails,
    generate_normalization_thumbnails,
    generate_segmentation_thumbnails,
    make_nipype_execution_log_html,
    )


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
                 session_id=None, output_dir=None, session_output_dirs=None,
                 anat_output_dir=None, **kwargs):
        self.func = func
        self.anat = anat
        self.subject_id = subject_id
        self.session_id = session_id
        self.output_dir = output_dir
        self.anat_output_dir = anat_output_dir
        self.session_output_dirs = session_output_dirs
        self.failed = False

        # nipype outputs
        self.nipype_results = {}

        self._set_items(**kwargs)

    def _set_items(self, **kwargs):
        for k, v in kwargs.iteritems():
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
        self.func = [mem.cache(delete_orientation)(
                self.func[j],
                self.tmp_output_dir,
                output_tag=self.session_id[j])
                     for j in xrange(len(self.session_id))]

        # deleteorient for anat
        if not self.anat is None:
            self.anat = mem.cache(delete_orientation)(
                self.anat, self.tmp_output_dir)

    def _sanitize_output_dir(self):

        # output dir
        assert not self.output_dir is None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # anat output dir
        if self.anat_output_dir is None:
            self.anat_output_dir = self.output_dir
        if not os.path.exists(self.anat_output_dir):
            os.makedirs(self.anat_output_dir)

        # func output dirs (one per session)
        if self.session_output_dirs is None:
            self.session_output_dirs = [None] * self.n_sessions

        for sess, sess_output_dir in enumerate(self.session_output_dirs):
            if sess_output_dir is None:
                if self.n_sessions > 1:
                    sess_output_dir = os.path.join(
                        self.output_dir, self.session_id[sess])
                else:
                    sess_output_dir = self.output_dir

            if not os.path.exists(sess_output_dir):
                os.makedirs(sess_output_dir)

            self.session_output_dirs[sess] = sess_output_dir

        # make tmp output dir
        self.tmp_output_dir = os.path.join(self.output_dir,
                                           "tmp")
        if not os.path.exists(self.tmp_output_dir):
            os.makedirs(self.tmp_output_dir)

    def _niigz2nii(self):
        """
        Convert .nii.gz to .nii (crucial for SPM).

        """

        cache_dir = os.path.join(self.output_dir, 'cache_dir')
        mem = Memory(cache_dir, verbose=100)

        self.func = [mem.cache(do_niigz2nii)(
                self.func[sess],
                output_dir=self.session_output_dirs[sess])
                     for sess in xrange(self.n_sessions)]

        if not self.anat is None:
            self.anat = mem.cache(do_niigz2nii)(self.anat,
                                                output_dir=self.output_dir)

    def _dcm2nii(self):
        """
        Convert DICOM to nifti.

        """

        self.isdicom = False
        if not isinstance(self.func[0], basestring):
            self.isdicom = isdicom(self.func[0][0])
        self.func = [do_dcm2nii(sess_func, output_dir=self.output_dir)[0]
                     for sess_func in self.func]

        if not self.anat is None:
            self.anat = do_dcm2nii(self.anat, output_dir=self.output_dir)[0]

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

        # sanitize func
        if isinstance(self.func, basestring):
            self.func = [self.func]

        # sanitize anat
        if not self.anat is None:
            assert os.path.isfile(self.anat)

        # sanitize session_id
        if self.session_id is None:
            if len(self.func) < 10:
                self.session_id = ["session_%i" % i
                                   for i in xrange(len(self.func))]
            else:
                self.session_id = ["session_0"]
        else:
            if isinstance(self.session_id, (basestring, int)):
                assert len(self.func) == 1
                self.session_id = [self.session_id]
            else:
                assert len(self.session_id) == len(self.func), "%s != %s" % (
                    self.session_id, len(self.func))
        self.n_sessions = len(self.session_id)

        # sanitize output_dir
        self._sanitize_output_dir()

        # .dcm, .ima -> .nii
        self._dcm2nii()

        if deleteorient:
            self._delete_orientation()

        # .nii.gz -> .nii extraction for SPM & co.
        if niigz2nii:
            self._niigz2nii()

        # XXX issue #40
        if len(set(self.func)) < len(self.func):
            raise RuntimeError(
                "Session func images must have unique abspaths; got %s" % (
                    self.func))

        return self

    def hardlink_output_files(self, final=False):
        """
        Hard-links output files to subject's immediate output directory.

        Parameters
        ----------
        final: bool, optional (default False)
            flag indicating whether, we're finalizing the preprocessing
            pipeline

        """

        # anat stuff
        for item in ["anat",
                     'gm', 'wm', 'csf',  # native
                     'wgm', 'wwm', 'wcsf'  # warped/normalized
                     ]:
            if hasattr(self, item):
                filename = getattr(self, item)
                if not filename is None:
                    linked_filename = hard_link(filename, self.anat_output_dir)
                    if final:
                        setattr(self, item, linked_filename)

        # func stuff
        for item in ['func', 'realignment_parameters']:
            tmp = []
            if hasattr(self, item):
                filenames = getattr(self, item)
                if isinstance(filenames, basestring):
                    assert self.n_sessions == 1
                    filenames = [filenames]
                for sess in xrange(self.n_sessions):
                    filename = filenames[sess]
                    if not filename is None:
                        linked_filename = hard_link(
                            filename, self.session_output_dirs[sess])
                        tmp.append(linked_filename)
            if final:
                setattr(self, item, tmp)

    def init_report(self, parent_results_gallery=None,
                    cv_tc=True, preproc_undergone=None):
        """
        This method is invoked to initialize the reports factory for the
        subject. It configures everything necessary for latter reporting:
        copies the custom .css, .js, etc. files, generates markup for
        the report pages (report.html, report_log.html, report_preproc.html,
        etc.), etc.

        Parameters
        ----------
        cv_tc: bool (optional)
            if set, a summarizing the time-course of the coefficient of
            variation in the preprocessed fMRI time-series will be
            generated

        """

        # make sure output_dir is OK
        self._sanitize_output_dir()

        # make separate dir for reports
        self.reports_output_dir = os.path.join(self.output_dir, "reports")
        if not os.path.exists(self.reports_output_dir):
            os.makedirs(self.reports_output_dir)

        self.report = True
        self.results_gallery = None
        self.parent_results_gallery = parent_results_gallery
        self.cv_tc = cv_tc

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
            results=self.results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            subject_id=self.subject_id,
            )
        main_html = get_subject_report_html_template(
            start_time=time.ctime(),
            subject_id=self.subject_id
            )

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

        if not self.reporting_enabled():
            return

        if parent_results_gallery is None:
            parent_results_gallery = self.parent_results_gallery

        # generate failure thumbnail
        if self.failed:
            copy_failed_png(self.reports_output_dir)
            self.final_thumbnail.img.src = 'failed.png'
            self.final_thumbnail.description += (
                ' (failed realignment)')
        else:
            # geneate cv_tc plots
            if self.cv_tc:
                generate_cv_tc_thumbnail(
                    self.func,
                    self.session_id,
                    self.subject_id,
                    self.reports_output_dir,
                    results_gallery=self.results_gallery
                    )

        # shut down all watched report pages
        self.progress_logger.finish_all()

        # shut down everything in reports_output_dir
        if last_stage:
            self.progress_logger.finish_dir(self.reports_output_dir)

        # commit final thumbnail into parent's results gallery
        if not parent_results_gallery is None:
            commit_subject_thumnbail_to_parent_gallery(
                self.final_thumbnail,
                self.subject_id,
                parent_results_gallery)

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

    def generate_realignment_thumbnails(self, log=True):
        """
        Invoked to generate post-realignment thumbnails.

        """

        if not hasattr(self, 'realignment_parameters'):
            print(
                "self has no field 'realignment_parameters'; nothing to do")
            return

        if not self.reporting_enabled():
            self.init_report()

        # log execution
        if log:
            execution_log_html = make_nipype_execution_log_html(
                    self.func, "Motion_Correction",
                    self.reports_output_dir)
            self.progress_logger.log(
                "<b>Motion Correction</b><br/>")
            self.progress_logger.log(open(execution_log_html).read())
            self.progress_logger.log('<hr/>')

        thumbs = generate_realignment_thumbnails(
            getattr(self, 'realignment_parameters'),
            self.reports_output_dir,
            sessions=self.session_id,
            execution_log_html_filename=execution_log_html if log
            else None,
            results_gallery=self.results_gallery
            )

        self.final_thumbnail.img.src = thumbs['rp_plot']

    def generate_coregistration_thumbnails(self, coreg_func_to_anat=True,
                                           log=True, comment=True):
        """
        Invoked to generate post-coregistration thumbnails.

        """

        # subject has anat ?
        if self.anat is None:
            print("Subject 'anat' field is None; nothing to do")
            return

        # reporting enabled ?
        if not self.reporting_enabled():
            self.init_report()

        src, ref = self.func, self.anat
        src_brain, ref_brain = "func", "anat"
        if not coreg_func_to_anat:
            src, ref = ref, src
            src_brain, ref_brain = ref_brain, src_brain

        # log execution
        if log:
            execution_log_html = make_nipype_execution_log_html(
                    src, "Coregister",
                    self.reports_output_dir)
            self.progress_logger.log(
                "<b>Coregistration</b><br/>")
            self.progress_logger.log(open(execution_log_html).read())
            self.progress_logger.log('<hr/>')

        # generate thumbs proper
        thumbs = generate_coregistration_thumbnails(
            (ref, ref_brain),
            (src, src_brain),
            self.reports_output_dir,
            execution_log_html_filename=execution_log_html if log
            else None,
            results_gallery=self.results_gallery,
            comment=comment
            )

        self.final_thumbnail.img.src = thumbs['axial']

    def generate_segmentation_thumbnails(self, log=True):
        """
        Invoked to generate post-segmentation thumbnails.

        """

        # segmentation done ?
        segmented = False
        for item in ['gm', 'wm', 'csf']:
            if hasattr(self, item):
                segmented = True
                break
        if not segmented:
            return

        # reporting enabled ?
        if not self.reporting_enabled():
            self.init_report()

        # log execution
        if log:
            execution_log_html = make_nipype_execution_log_html(
                getattr(self, 'gm') or getattr(self, 'wm') or getattr(
                    self, 'csf'), "Segment", self.reports_output_dir)
            self.progress_logger.log(
                "<b>Segmentation</b><br/>")
            self.progress_logger.log(open(execution_log_html).read())
            self.progress_logger.log('<hr/>')

        for brain_name, brain, cmap in zip(
            ['anat', 'func'], [self.anat, self.func],
            [cm.gray, cm.spectral]):
            thumbs = generate_segmentation_thumbnails(
                brain,
                self.reports_output_dir,
                subject_gm_file=getattr(self, 'gm', None),
                subject_wm_file=getattr(self, 'wm', None),
                subject_csf_file=getattr(self, 'csf', None),
                cmap=cmap,
                brain=brain_name,
                only_native=True,
                execution_log_html_filename=execution_log_html if log
                else None,
                results_gallery=self.results_gallery
                )

            if brain_name == 'func':
                self.final_thumbnail.img.src = thumbs['axial']

    def generate_normalization_thumbnails(self, log=True):
        """
        Invoked to generate post-normalization thumbnails.

        """

        # reporting enabled ?
        if not self.reporting_enabled():
            self.init_report()

        # segmentation done ?
        segmented = False
        for item in ['wgm', 'wwm', 'wcsf']:
            if hasattr(self, item):
                segmented = True
                break

        for brain_name, brain, cmap in zip(
            ['anat', 'func'], [self.anat, self.func],
            [cm.gray, cm.spectral]):

            if not hasattr(self, brain_name):
                continue

            # generate segmentation thumbs
            if segmented:
                thumbs = generate_segmentation_thumbnails(
                    brain,
                    self.reports_output_dir,
                    subject_gm_file=getattr(self, 'wgm', None),
                    subject_wm_file=getattr(self, 'wwm', None),
                    subject_csf_file=getattr(self, 'wcsf', None),
                    cmap=cmap,
                    brain=brain_name,
                    comments="warped",
                    execution_log_html_filename=make_nipype_execution_log_html(
                        getattr(self, 'wgm') or getattr(
                            self, 'wwm') or getattr(
                            self, 'wcsf'), "Segment",
                        self.reports_output_dir) if log else None,
                    results_gallery=self.results_gallery
                    )

                if brain_name == 'func':
                    self.final_thumbnail.img.src = thumbs['axial']

            # log execution
            if log:
                execution_log_html = make_nipype_execution_log_html(
                    brain, "Normalization", self.reports_output_dir)
                self.progress_logger.log(
                    "<b>Normalization of %s</b><br/>" % brain_name)
                text = open(execution_log_html).read()
                if "normalized_files" in text or "warped_files" in text:
                    self.progress_logger.log(text)
                self.progress_logger.log('<hr/>')

            # generate normalization thumbs proper
            thumbs = generate_normalization_thumbnails(
                brain,
                self.reports_output_dir,
                brain=brain_name,
                execution_log_html_filename=execution_log_html if log
                else None,
                results_gallery=self.results_gallery,
                )

            if not segmented and brain_name == 'func':
                self.final_thumbnail.img.src = thumbs['axial']

    def generate_smooth_thumbnails(self):
        """
        Generate thumbnails post-smoothing.

        """

        # reporting enabled ?
        if not self.reporting_enabled():
            self.init_report()

        # log execution
        execution_log_html = make_nipype_execution_log_html(
                self.func, "Smooth",
                self.reports_output_dir)
        self.progress_logger.log(
            "<b>Smooth</b><br/>")
        text = open(execution_log_html).read()
        if "smoothed_files" in text:
            self.progress_logger.log(text)
        self.progress_logger.log('<hr/>')

    def generate_report(self, **kwargs):
        """
        Method invoked to generate all reports in one-go. This is useful
        for generating reports out-side the preproc logic: simply populate
        the func, anat (optionally realignment_params, gm, wm, csf, etc.)
        fields and then fire this method.

        """

        # set items
        self.set_items()

        # sanitiy
        self.sanitize()

        # report
        self.generate_realignment_thumbnails(log=False)
        self.generate_coregistration_thumbnails(log=False, comment=False)
        self.generate_normalization_thumbnails(log=False)

        # finalize the business
        self.finalize_report(last_stage=True)
