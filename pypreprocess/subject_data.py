"""
:Synopsis: Encapsulation of subject data. Handles subject data logic
like filetypes, extensions, .nii.gz -> .nii conversions, setup and
teardown for reports, etc., general sanitization, etc.
:Author: DOHMATOB Elvis Dopgima

"""

import os
import time
from joblib import Memory
from .io_utils import (niigz2nii,
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
    get_subject_report_preproc_html_template
    )
from .reporting.preproc_reporter import generate_cv_tc_thumbnail


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
                 session_id=None, output_dir=None, **kwargs):
        self.func = func
        self.anat = anat
        self.subject_id = subject_id
        self.session_id = session_id
        self.output_dir = output_dir
        self.failed = False

        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def delete_orientation(self):
        # prepare for smart caching
        cache_dir = os.path.join(self.output_dir, 'cache_dir')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        mem = Memory(cachedir=cache_dir, verbose=5)

        # deleteorient for func
        self.func = [mem.cache(delete_orientation)(
                self.func[j],
                self.output_dir,
                output_tag=self.session_id[j])
                     for j in xrange(len(self.session_id))]

        # deleteorient for anat
        print self.anat
        if not self.anat is None:
            self.anat = mem.cache(delete_orientation)(
                self.anat, self.output_dir)

    def sanitize(self, do_deleteorient=False, do_niigz2nii=False):
        """
        This method does basic sanitization of the `SubjectData` instance, like
        extracting .nii.gz -> .nii (crusial for SPM), ensuring that functional
        images actually exist on disk, etc.

        Parameters
        ----------
        do_deleteorient: bool (optional)
            if true, then orientation meta-data in all input image files
            for this subject will be stripped-off

        do_niigz2nii: bool, optional (default False)
            convert func and ant .nii.gz images to .nii

        """

        # sanitize output_dir
        assert not self.output_dir is None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # sanitize func
        if isinstance(self.func, basestring):
            self.func = [self.func]

        # sanitize anat
        if not self.anat is None:
            assert os.path.isfile(self.anat)

        # .nii.gz -> .nii extraction for SPM & co.
        if do_niigz2nii:
            cache_dir = os.path.join(self.output_dir, 'cache_dir')
            mem = Memory(cache_dir, verbose=100)

            self.func = mem.cache(niigz2nii)(self.func,
                                             output_dir=self.output_dir)
            if not self.anat is None:
                self.anat = mem.cache(niigz2nii)(self.anat,
                                                 output_dir=self.output_dir)

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

        if do_deleteorient:
            self.delete_orientation()

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

        for item in ['func', 'anat', 'realignment_parameters',
                     'gm', 'wm', 'csf',  # native
                     'wgm', 'wwm', 'wcsf'  # warped/normalized
                     ]:
            if hasattr(self, item):
                filename = getattr(self, item)
                if not filename is None:
                    linked_filename = hard_link(filename, self.output_dir)
                    if final:
                        setattr(self, item, linked_filename)

    def init_report(self, parent_results_gallery=None, last_stage=False,
                    do_cv_tc=True, preproc_undergone=None):
        self.do_report = True
        self.results_gallery = None
        self.parent_results_gallery = parent_results_gallery
        self.last_stage = last_stage
        self.do_cv_tc = do_cv_tc

        # report filenames
        self.report_log_filename = os.path.join(
            self.output_dir, 'report_log.html')
        self.report_preproc_filename = os.path.join(
            self.output_dir, 'report_preproc.html')
        self.report_filename = os.path.join(self.output_dir,
                                                    'report.html')

        # initialize results gallery
        loader_filename = os.path.join(self.output_dir,
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
        copy_web_conf_files(self.output_dir)

        # initialize progress bar
        self.progress_logger = ProgressReport(
            self.report_log_filename,
            other_watched_files=[self.report_preproc_filename])

        # html markup
        preproc = get_subject_report_preproc_html_template(
            ).substitute(
            results=self.results_gallery,
            start_time=time.ctime(),
            preproc_undergone=preproc_undergone,
            subject_id=self.subject_id,
            )
        main_html = get_subject_report_html_template(
            ).substitute(
            start_time=time.ctime(),
            subject_id=self.subject_id
            )

        with open(self.report_preproc_filename, 'w') as fd:
            fd.write(str(preproc))
            fd.close()
        with open(self.report_filename, 'w') as fd:
            fd.write(str(main_html))
            fd.close()

    def _finalize_report(self):
        """
        Finalizes the business of reporting.

        """

        if not self.do_report:
            return

        # generate failure thumbnail
        if self.failed:
            self.final_thumbnail.img.src = 'failed.png'
            self.final_thumbnail.description += (
                ' (failed realignment)')
        else:
            # geneate cv_tc plots
            if self.do_cv_tc:
                generate_cv_tc_thumbnail(
                    self.func,
                    self.session_id,
                    self.subject_id,
                    self.output_dir,
                    results_gallery=self.results_gallery
                    )

        if self.last_stage:
            self.progress_logger.finish_dir(self.output_dir)

        if not self.parent_results_gallery is None:
            commit_subject_thumnbail_to_parent_gallery(
                self.final_thumbnail,
                self.subject_id,
                self.parent_results_gallery)

        print ("\r\nPreproc report for subject %s written to %s"
               " .\r\n" % (self.subject_id,
                           self.report_preproc_filename))

    def __getitem__(self, key):
        return self.__dict__[key]
