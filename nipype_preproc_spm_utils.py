"""
:Module: nipype_preproc_spm_utils
:Synopsis: routine functions for SPM preprocessing business
:Author: dohmatob elvis dopgima (hereafter referred to as DED)

XXX TODO: document the code!
XXX TODO: re-factor the code!
"""

# standard imports
import os
import shutil

# imports for caching (yeah, we aint got time to loose!)
from nipype.caching import Memory

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

# set job count
N_JOBS = -1
if 'N_JOBS' in os.environ:
    N_JOBS = int(os.environ['N_JOBS'])

# set templates
T1_TEMPLATE = os.path.join(SPM_DIR, 'templates/T1.nii')
GM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/grey.nii')
WM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/white.nii')
CSF_TEMPLATE = os.path.join(SPM_DIR, 'tpm/csf.nii')


class SubjectData(object):
    """
    Encapsulation for subject data, relative to preprocessing.

    """

    pass


def do_subject_realign(output_dir,
                       subject_id=None,
                       do_report=True,
                       **spm_realign_kwargs):
    output = {}

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
        import check_preprocessing
        import reporter

        rp = realign_result.outputs.realignment_parameters

        output['thumbnails'] = []

        rp_plot = os.path.join(output_dir, 'rp_plot.png')
        check_preprocessing.plot_spm_motion_parameters(
        rp,
        subject_id=subject_id,
        title="Plot of motion parameters before realignment",
        output_filename=rp_plot)

        # nipype report
        nipype_report_filename = os.path.join(
            os.path.dirname(rp),
            "_report/report.rst")
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = reporter.nipype2htmlreport(nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        # create thumbnail
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(href=os.path.basename(rp_plot))
        thumbnail.img = reporter.img(src=os.path.basename(rp_plot),
                                     height="500px",
                                     width="1200px")
        thumbnail.description = \
            "Motion Correction (<a href=%s>see execution log</a>)" \
            % nipype_html_report_filename
        output['thumbnails'].append(thumbnail)

    # collect ouput
    output['result'] = realign_result
    output['rp_plot'] = rp_plot

    return output


def do_subject_coreg(output_dir,
                     subject_id=None,
                     do_report=True,
                     comments="",
                     **spm_coregister_kwargs):
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
        import check_preprocessing
        import reporter
        import pylab as pl

        output['thumbnails'] = []

        # nipype report
        nipype_report_filename = os.path.join(
            os.path.dirname(coreg_result.outputs.coregistered_source),
            "_report/report.rst")
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = reporter.nipype2htmlreport(nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        # prepare for smart caching
        qa_cache_dir = os.path.join(output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

        # plot overlap of target on coregistered source
        target = spm_coregister_kwargs['target']
        source = coreg_result.outputs.coregistered_source

        overlap = os.path.join(
            output_dir,
            "%s_on_%s_overlap.png" % (os.path.basename(target),
                                      os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=overlap,
            cmap=pl.cm.gray,
            title="Overlap of %s on %s" % (os.path.basename(target),
                                           os.path.basename(source)))

        overlap_axial = os.path.join(
            output_dir,
            "%s_on_%s_overlap_axial.png" % (os.path.basename(target),
                                            os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=overlap_axial,
            slicer='z',
            cmap=pl.cm.gray,
            title="%s: coreg" % subject_id)

        # create thumbnail
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(href=os.path.basename(overlap))
        thumbnail.img = reporter.img(src=os.path.basename(overlap),
                                     height="500px")
        thumbnail.description = \
            "Coregistration %s (<a href=%s>see execution log</a>)" \
            % (comments, nipype_html_report_filename)

        output['thumbnails'].append(thumbnail)

        output['axial_overlap'] = overlap_axial

        # plot overlap of coregistered source on target
        source, target = (target, source)
        overlap = os.path.join(
            output_dir,
            "%s_on_%s_overlap.png" % (os.path.basename(target),
                                      os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=overlap,
            cmap=pl.cm.gray,
            title="Overlap of %s on %s" % (os.path.basename(target),
                                           os.path.basename(source)))

        overlap_axial = os.path.join(
            output_dir,
            "%s_on_%s_overlap_axial.png" % (os.path.basename(target),
                                            os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=overlap_axial,
            cmap=pl.cm.gray,
            slicer='z',
            title="%s: coreg" % subject_id)

        # create thumbnail
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(href=os.path.basename(overlap))
        thumbnail.img = reporter.img(src=os.path.basename(overlap),
                                     height="500px")
        thumbnail.description = \
            "Coregistration %s (<a href=%s>see execution log</a>)" \
            % (comments, nipype_html_report_filename)
        output['thumbnails'].append(thumbnail)

    # collect ouput
    output['result'] = coreg_result

    return output


def do_subject_segment(output_dir,
                       subject_id=None,
                       do_report=True,
                       **spm_segment_kwargs):
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
        pass

    # collect ouput
    output['result'] = segment_result

    return output


def do_subject_normalize(output_dir,
                         subject_id=None,
                         do_report=True,
                         segment_result=None,
                         brain="brain",
                         cmap=None,
                         **spm_normalize_kwargs):
    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = Memory(base_dir=cache_dir)

    # run workflow
    normalize = mem.cache(spm.Normalize)
    norm_result = normalize(**spm_normalize_kwargs)

    # generate gallery for HTML report
    if do_report and not segment_result is None:
        import nibabel
        import check_preprocessing
        import reporter
        import pylab as pl

        # prepare for smart caching
        qa_cache_dir = os.path.join(output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

        # nipype report
        nipype_report_filename = os.path.join(
            os.path.dirname(norm_result.outputs.normalized_files),
            "_report/report.rst")
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = reporter.nipype2htmlreport(
            nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        output['thumbnails'] = []

        normalized_file = norm_result.outputs.normalized_files

        # /!\
        # If the normalized data is 4D, then do QA on it's mean (along)
        # (the time axis) instead; if we did well, then the mean should
        # align with the template gm, wm, csf compartments pretty well,
        # and with the MNI template too; else, we've got a failed
        # normalization.
        normalized_img = nibabel.load(normalized_file)
        if len(normalized_img.shape) == 4:
            mean_normalized_img = nibabel.Nifti1Image(
                normalized_img.get_data().mean(-1),
                normalized_img.get_affine())
            mean_normalized_file = os.path.join(
                os.path.dirname(normalized_file),
                'mean' + os.path.basename(
                    normalized_file))
            nibabel.save(mean_normalized_img, mean_normalized_file)
            normalized_file = mean_normalized_file

        # plot contours of template compartments on subject's brain
        template_compartments_contours = os.path.join(
            output_dir,
            "template_tmps_contours_on_%s.png" % brain)
        template_compartments_contours_axial = os.path.join(
            output_dir,
            "template_compartments_contours_on_%s_axial.png" % brain)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            normalized_file,
            GM_TEMPLATE,
            WM_TEMPLATE,
            CSF_TEMPLATE,
            output_filename=template_compartments_contours_axial,
            slicer='z',
            cmap=cmap,
            title="template TPMs")

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            normalized_file,
            GM_TEMPLATE,
            WM_TEMPLATE,
            CSF_TEMPLATE,
            output_filename=template_compartments_contours,
            cmap=cmap,
            title="Template GM, WM, and CSF contours on subject's %s" % \
                brain)

        # create thumbnail
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(
            href=os.path.basename(template_compartments_contours))
        thumbnail.img = reporter.img(
            src=os.path.basename(template_compartments_contours),
            height="500px")
        thumbnail.description \
            = "Normalization (<a href=%s>see execution log</a>)" \
            % (nipype_html_report_filename)
        output['thumbnails'].append(thumbnail)

        # plot contours of subject's compartments on subject's brain
        subject_compartments_contours = os.path.join(
            output_dir,
            "subject_tmps_contours_on_subject_%s.png" % brain)
        subject_compartments_contours_axial = os.path.join(
            output_dir,
            "subject_tmps_contours_on_subject_%s_axial.png" % brain)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            normalized_file,
            segment_result.outputs.modulated_gm_image,
            segment_result.outputs.modulated_wm_image,
            segment_result.outputs.modulated_csf_image,
            output_filename=subject_compartments_contours_axial,
            slicer='z',
            cmap=cmap,
            title="subject TPMs")

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            normalized_file,
            segment_result.outputs.modulated_gm_image,
            segment_result.outputs.modulated_wm_image,
            segment_result.outputs.modulated_csf_image,
            output_filename=subject_compartments_contours,
            cmap=cmap,
            title="Subject's GM, WM, and CSF contours on subject's %s" % \
                brain)

        # create thumbnail
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(
            href=os.path.basename(subject_compartments_contours))
        thumbnail.img = reporter.img(
            src=os.path.basename(subject_compartments_contours),
            height="500px")
        thumbnail.description = \
            "Normalization (<a href=%s>see execution log</a>)" \
            % nipype_html_report_filename
        output['thumbnails'].append(thumbnail)

        # check registration
        target = os.path.join(SPM_DIR, "templates/T1.nii")
        source = normalized_file

        # plot overlap (edge map) of MNI template on the
        # normalized image
        overlap = os.path.join(
            output_dir,
            "%s_on_%s_overlap.png" % (os.path.basename(target),
                                      os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=overlap,
            cmap=cmap,
            title="Overlap of %s on %s" % (os.path.basename(target),
                                           os.path.basename(source)))

        # create thumbnail
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(href=os.path.basename(overlap))
        thumbnail.img = reporter.img(
            src=os.path.basename(overlap), height="500px")
        thumbnail.description = \
            "Normalization (<a href=%s>see execution log</a>)" \
            % nipype_html_report_filename
        output['thumbnails'].append(thumbnail)

        # plot overlap (edge map) of the normalized image
        # on the MNI template
        source, target = (target, source)
        overlap = os.path.join(
            output_dir,
            "%s_on_%s_overlap.png" % (os.path.basename(target),
                                      os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=overlap,
            cmap=pl.cm.gray,
            title="Overlap of %s on %s" % (os.path.basename(target),
                                           os.path.basename(source)))

        # create thumbnail
        thumbnail = reporter.Thumbnail()
        thumbnail.a = reporter.a(href=os.path.basename(overlap))
        thumbnail.img = reporter.img(
            src=os.path.basename(overlap), height="500px")
        thumbnail.description = \
            "Normalization (<a href=%s>see execution log</a>)" \
            % nipype_html_report_filename
        output['thumbnails'].append(thumbnail)

        output['axials'] = {}
        output['axials']['template_compartments_contours'] = \
            template_compartments_contours_axial
        output['axials']['subject_compartments_contours'] = \
            subject_compartments_contours_axial

    # collect ouput
    output['result'] = norm_result

    return output


def do_subject_preproc(
    subject_data,
    do_report=True,
    do_bet=True,
    do_slicetiming=False,
    do_realign=True,
    do_coreg=True,
    do_segment=True,
    do_cv_tc=True,
    parent_results_gallery=None,
    main_page="#"):
    """
    Function preprocessing data for a single subject.

    """

    if not os.path.exists(subject_data.output_dir):
        os.makedirs(subject_data.output_dir)

    if do_report:
        import check_preprocessing
        import reporter

        report_filename = os.path.join(subject_data.output_dir, "_report.html")
        final_thumbnail = reporter.Thumbnail()
        final_thumbnail.a = reporter.a(href=report_filename)
        final_thumbnail.img = reporter.img()
        final_thumbnail.description = subject_data.subject_id

        # initialize results gallery
        loader_filename = os.path.join(
            subject_data.output_dir, "results_loader.php")
        results_gallery = reporter.ResultsGallery(
            loader_filename=loader_filename,
            title="Report for subject %s" % subject_data.subject_id)

        # html markup
        report = reporter.SUBJECT_PREPROC_REPORT_HTML_TEMPLATE().substitute(
            results=results_gallery,
            main_page=main_page)

        print ">" * 80 + "BEGIN HTML"
        print report
        print "<" * 80 + "END HTML"

        with open(report_filename, 'w') as fd:
            fd.write(str(report))
            fd.close()

    # final fMRI images (re-set after each stage/node)
    final_func = subject_data.func

    # brain extraction (bet)
    if do_bet:
        pass

    #####################
    #  motion correction
    #####################
    if do_realign:
        realign_output = do_subject_realign(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=do_report,
            in_files=subject_data.func,
            register_to_mean=True,
            jobtype='estwrite',
            )

        # collect output
        realign_result = realign_output['result']
        final_func = realign_result.outputs.realigned_files
        mean_func = realign_result.outputs.mean_image

        # generate report stub
        if do_report:
            results_gallery.commit_thumbnails(realign_output['thumbnails'])
            final_thumbnail.img.src = realign_output['rp_plot']
    else:
        import nibabel

        # manually compute mean (along time axis) of fMRI images
        func_images = nibabel.load(final_func)
        mean_func_image = nibabel.Nifti1Image(
            func_images.get_data().mean(-1), func_images.get_affine())
        mean_func = os.path.join(
            os.path.dirname(subject_data.func),
            'mean' + os.path.basename(subject_data.func))
        nibabel.save(mean_func_image, mean_func)

    ################################################################
    # co-registration of structural (anatomical) against functional
    ################################################################
    if do_coreg:
        # specify input files for coregistration
        coreg_target = mean_func
        coreg_source = subject_data.anat

        # run coreg proper
        coreg_output = do_subject_coreg(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=do_report,
            comments="anat -> epi",
            target=coreg_target,
            source=coreg_source,
            jobtype='estimate',
            )

        # collect results
        coreg_result = coreg_output['result']

        # rest anat to coregistered version thereof
        subject_data.anat = coreg_result.outputs.coregistered_source

        # generate report stub
        if do_report:
            results_gallery.commit_thumbnails(coreg_output['thumbnails'])
            final_thumbnail.img.src = coreg_output['axial_overlap']

    ###################################
    # segmentation of anatomical image
    ###################################
    if do_segment:
        segment_data = subject_data.anat
        segment_output = do_subject_segment(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=False,  # XXX why ?
            data=segment_data,
            gm_output_type=[True, True, True],  # XXX why ?
            wm_output_type=[True, True, True],  # XXX why ?
            csf_output_type=[True, True, True],  # XXX why ?
            tissue_prob_maps=[GM_TEMPLATE,
                              WM_TEMPLATE, CSF_TEMPLATE],
            # the following kwargs are courtesy of christophe p.
            gaussians_per_class=[2, 2, 2, 4],
            affine_regularization="mni",
            bias_regularization=0.0001,
            bias_fwhm=60,
            warping_regularization=1,
            )

        segment_result = segment_output['result']

        ##############################################################
        # indirect normalization: warp fMRI images int into MNI space
        # using the deformations learned by segmentation
        ##############################################################
        norm_parameter_file = segment_result.outputs.transformation_mat
        norm_apply_to_files = final_func

        norm_output = do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            segment_result=segment_result,
            do_report=True,
            brain='epi',
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=[3, 3, 3],
            write_interp=1,
            jobtype='write',
            )

        if do_report:
            import pylab as pl  # we need color maps
            results_gallery.commit_thumbnails(norm_output['thumbnails'])
            final_thumbnail.img.src = \
                norm_output['axials']["template_compartments_contours"]

        #############################################################
        # indirect normalization: warp anat image int into MNI space
        # using the deformations learned by segmentation
        #############################################################
        norm_parameter_file = segment_result.outputs.transformation_mat
        norm_apply_to_files = subject_data.anat

        norm_output = do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            segment_result=segment_result,
            brain="anat",
            cmap=pl.cm.gray,
            do_report=do_report,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=[1, 1, 1],
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write')

        norm_result = norm_output['result']

        if do_report:
            results_gallery.commit_thumbnails(norm_output['thumbnails'])

        # normalize func images based on the learned segmentation
        norm_parameter_file = segment_result.outputs.transformation_mat
        norm_apply_to_files = final_func

        norm_output = do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=False,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            jobtype='write',
            )

        norm_result = norm_output['result']
        final_func = norm_result.outputs.normalized_files

    # generate html report (for QA)
    if do_report:
        blablabla = "Generating QA reports for subject %s .."\
            % subject_data.subject_id
        dadada = "+" * len(blablabla)
        print "\r\n%s\r\n%s\r\n%s\r\n" % (dadada, blablabla, dadada)

        if do_cv_tc:
            qa_cache_dir = os.path.join(subject_data.output_dir, "QA")
            if not os.path.exists(qa_cache_dir):
                os.makedirs(qa_cache_dir)
            qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

            cv_tc_plot_after = os.path.join(
                subject_data.output_dir, "cv_tc_after.png")

            corrected_FMRIs = [final_func]
            qa_mem.cache(
                check_preprocessing.plot_cv_tc)(
                corrected_FMRIs, [subject_data.session_id],
                subject_data.subject_id,
                subject_data.output_dir,
                cv_tc_plot_outfile=cv_tc_plot_after,
                plot_diff=True,
                title="")

            # create thumbnail
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(href=cv_tc_plot_after)
            thumbnail.img = reporter.img(
                src=os.path.basename(cv_tc_plot_after), height="500px",
                width="1200px")
            thumbnail.description = "Coefficient of Variation"
            results_gallery.commit_thumbnails(thumbnail)

        final_thumbnail.img.height = "250px"
        final_thumbnail.img.src = "%s/%s/%s" % (
            subject_data.session_id,
            subject_data.subject_id,
            os.path.basename(final_thumbnail.img.src))
        final_thumbnail.a.href = "%s/%s/%s" % (
            subject_data.session_id,
            subject_data.subject_id,
            os.path.basename(final_thumbnail.a.href))

        if parent_results_gallery:
            parent_results_gallery.commit_thumbnails(final_thumbnail)


def do_group_preproc(subjects,
                     do_report=True,
                     do_export_report=True,
                     dataset_description=None,
                     report_filename=None,
                     do_bet=False,
                     do_realign=True,
                     do_coreg=True,
                     do_segment=True,
                     do_cv_tc=True):

    """
    This functions doe intra-subject fMRI preprocessing on a
    group os subjects.

    Parameters
    ==========
    subjects: iterable of SubjectData objects

    report_filename: string (optional)
    if provided, an HTML report will be produced. This report is
    dynamic and its contents are updated automatically as more
    and more subjects are preprocessed.

    """

    # sanitize input
    if do_report:
        if report_filename is None:
            raise RuntimeError(
                ("You asked for reporting (do_report=True)  but specified"
                 " an invalid report_filename (None)"))

    kwargs = {'do_realign': do_realign, 'do_coreg': do_coreg,
                'do_segment': do_segment, 'do_cv_tc': do_cv_tc}

    # generate html report (for QA) as desired
    if do_report:
        import reporter

        # do some sanity
        shutil.copy(
            "css/styles.css", os.path.dirname(report_filename))

        # compute docstring explaining preproc steps undergone
        preproc_undergone = """\
<p>All preprocessing has been done using nipype's interface to the \
<a href="http://www.fil.ion.ucl.ac.uk/spm/">SPM8
package</a>.</p>"""

        step = 0

        if do_bet:
            step += 1
            preproc_undergone += (
                "%i. Brain extraction has been applied to strip-off the skull"
                " and other non-brain components from the subject's "
                "anatomical image. This prevents later registration problems "
                "like the skull been (mis-)aligned unto the cortical surface, "
                "etc.<br/>" % step)
        if do_realign:
            step += 1
            preproc_undergone += (
                "%i. Motion correction has been done so as to detect artefacts"
                " due to the subject's head motion during the acquisition, "
                "after which the images have been resliced.<br/>" % step)
        if do_coreg:
            step += 1
            preproc_undergone += (
                "%i. The subject's anatomical image has been coregistered "
                "against their fMRI images (precisely, to the mean thereof). "
                "Coregistration is important as it allows deformations of the "
                "anatomy (learned by registration against other images, "
                "templates for example) to be directly applicable to the fMRI."
                "<br/>" % step)
        if do_segment:
            step += 1
            preproc_undergone += (
                "%i. The anatomical image has been segmented into GM, WM,"
                " and CSF compartments by using TPMs (Tissue Probability Maps)"
                " as priors. The segmented anatomical image has been warped "
                "into the MNI template space by applying the deformations "
                "learned during segmentation. The same deformations have been"
                " applied to the fMRI images. This procedure is referred to "
                "as <i>indirect Normalization</i> in SPM jargon." % step)

        # initialize code for reporter from template
        tmpl = reporter.DATASET_PREPROC_REPORT_HTML_TEMPLATE()

        # prepare meta results
        loader_filename = os.path.join(os.path.dirname(report_filename),
                                       "results_loader.php")
        results_gallery = reporter.ResultsGallery(
            loader_filename=loader_filename)

        # write initial content for reporting
        report = tmpl.substitute(
            preproc_undergone=preproc_undergone,
            dataset_description=dataset_description,
            results=results_gallery)
        print ">" * 80 + "BEGIN HTML"
        print report
        print "<" * 80 + "END HTML\r\n"

        with open(report_filename, 'w') as fd:
            fd.write(str(report))
            fd.close()
            print "HTML report (dynamic) written to %s" % report_filename

    # preproc subjects
    shutil.copy('css/styles.css', "/tmp/styles.css")
    kwargs['parent_results_gallery'] = results_gallery
    kwargs['main_page'] = "../../%s" % os.path.basename(report_filename)

    joblib.Parallel(n_jobs=N_JOBS, verbose=100)(joblib.delayed(
            do_subject_preproc)(
            subject_data, **kwargs) for subject_data in subjects)

    print "HTML report (dynamic) written to %s" % report_filename

    # export report (so it can be emailed, for example)
    if do_report:
        if do_export_report:
            reporter.export_report(os.path.dirname(report_filename))
