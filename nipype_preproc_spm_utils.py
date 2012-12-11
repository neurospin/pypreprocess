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

# imports for nifti manip
import nibabel as ni

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
EPI_TEMPLATE = os.path.join(SPM_DIR, 'templates/EPI.nii')
T1_TEMPLATE = os.path.join(SPM_DIR, 'templates/T1.nii')
GM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/grey.nii')
WM_TEMPLATE = os.path.join(SPM_DIR, 'tpm/white.nii')
CSF_TEMPLATE = os.path.join(SPM_DIR, 'tpm/csf.nii')


class SubjectData(object):
    """
    Encapsulation for subject data, relative to preprocessing.

    XXX Why not inherit from Bunch ?
    """

    def __init__(self):
        self.subject_id = "subXYZ"
        self.session_id = "UNKNOWN_SESSION"
        self.anat = None
        self.func = None

    def delete_orientation(self):
        pass


def get_vox_dims(volume):
    """
    Infer voxel dimensions.

    """

    if isinstance(volume, list):
        volume = volume[0]
    nii = ni.load(volume)
    hdr = nii.get_header()
    voxdims = hdr.get_zooms()

    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]


def do_subject_realign(output_dir,
                       subject_id=None,
                       do_report=True,
                       results_gallery=None,
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
        if results_gallery:
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(href=os.path.basename(rp_plot))
            thumbnail.img = reporter.img(src=os.path.basename(rp_plot),
                                         height="500px",
                                         width="1200px")
            thumbnail.description = \
                "Motion Correction (<a href=%s>see execution log</a>)" \
                % nipype_html_report_filename

            results_gallery.commit_thumbnails(thumbnail)

    # collect ouput
    output['result'] = realign_result
    output['rp_plot'] = rp_plot

    return output


def do_subject_coreg(output_dir,
                     subject_id=None,
                     do_report=True,
                     results_gallery=None,
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

        # plot outline of target on coregistered source
        target = spm_coregister_kwargs['target']
        source = coreg_result.outputs.coregistered_source

        outline = os.path.join(
            output_dir,
            "%s_on_%s_outline.png" % (os.path.basename(target),
                                      os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline,
            cmap=pl.cm.gray,
            title="Outline of %s on %s" % (os.path.basename(target),
                                           os.path.basename(source)))

        outline_axial = os.path.join(
            output_dir,
            "%s_on_%s_outline_axial.png" % (os.path.basename(target),
                                            os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline_axial,
            slicer='z',
            cmap=pl.cm.gray,
            title="%s: coreg" % subject_id)

        # create thumbnail
        if results_gallery:
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(href=os.path.basename(outline))
            thumbnail.img = reporter.img(src=os.path.basename(outline),
                                         height="500px")
            thumbnail.description = \
                "Coregistration %s (<a href=%s>see execution log</a>)" \
                % (comments, nipype_html_report_filename)

            results_gallery.commit_thumbnails(thumbnail)

        output['axial_outline'] = outline_axial

        # plot outline of coregistered source on target
        source, target = (target, source)
        outline = os.path.join(
            output_dir,
            "%s_on_%s_outline.png" % (os.path.basename(target),
                                      os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline,
            cmap=pl.cm.gray,
            title="Outline of %s on %s" % (os.path.basename(target),
                                           os.path.basename(source)))

        outline_axial = os.path.join(
            output_dir,
            "%s_on_%s_outline_axial.png" % (os.path.basename(target),
                                            os.path.basename(source)))
        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline_axial,
            cmap=pl.cm.gray,
            slicer='z',
            title="%s: coreg" % subject_id)

        # create thumbnail
        if results_gallery:
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(href=os.path.basename(outline))
            thumbnail.img = reporter.img(src=os.path.basename(outline),
                                         height="500px")
            thumbnail.description = \
                "Coregistration %s (<a href=%s>see execution log</a>)" \
                % (comments, nipype_html_report_filename)
            results_gallery.commit_thumbnails(thumbnail)

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
                         results_gallery=None,
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
    if do_report:
        import check_preprocessing
        import reporter
        import pylab as pl

        normalized_files = norm_result.outputs.normalized_files
        if type(normalized_files) is str:
            normalized_files = [normalized_files]

        # prepare for smart caching
        qa_cache_dir = os.path.join(output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblib.Memory(cachedir=qa_cache_dir, verbose=5)

        # nipype report
        nipype_report_filename = os.path.join(
            os.path.dirname(normalized_files[0]),
            "_report/report.rst")
        nipype_html_report_filename = nipype_report_filename + '.html'
        nipype_report = reporter.nipype2htmlreport(
            nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        #####################
        # check registration
        #####################

        # plot outline (edge map) of MNI template on the
        # normalized image
        target = os.path.join(SPM_DIR, "templates/T1.nii")
        source = normalized_files

        outline = os.path.join(
            output_dir,
            "%s_on_%s_outline.png" % (os.path.basename(target),
                                      brain))

        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline,
            cmap=pl.cm.gray,
            title="Outline of MNI %s template on %s" % (
                os.path.basename(target),
                brain))

        # create thumbnail
        if results_gallery:
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(href=os.path.basename(outline))
            thumbnail.img = reporter.img(
                src=os.path.basename(outline), height="500px")
            thumbnail.description = \
                "Normalization (<a href=%s>see execution log</a>)" \
                % nipype_html_report_filename

            results_gallery.commit_thumbnails(thumbnail)

        # plot outline (edge map) of the normalized image
        # on the MNI template
        source, target = (target, source)
        outline = os.path.join(
            output_dir,
            "%s_on_%s_outline.png" % (brain,
                                      os.path.basename(source)))
        outline_axial = os.path.join(
            output_dir,
            "%s_on_%s_outline_axial.png" % (brain,
                                            os.path.basename(source)))

        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline_axial,
            slicer='z',
            cmap=cmap,
            title="Outline of %s on MNI %s template" % (
                brain,
                os.path.basename(source)))

        output['axial'] = outline_axial

        qa_mem.cache(check_preprocessing.plot_registration)(
            target,
            source,
            output_filename=outline,
            cmap=pl.cm.gray,
            title="Outline of %s on MNI %s template" % (
                brain,
                os.path.basename(source)))

        # create thumbnail
        if results_gallery:
            thumbnail = reporter.Thumbnail()
            thumbnail.a = reporter.a(href=os.path.basename(outline))
            thumbnail.img = reporter.img(
                src=os.path.basename(outline), height="500px")
            thumbnail.description = \
                "Normalization (<a href=%s>see execution log</a>)" \
                % nipype_html_report_filename

            results_gallery.commit_thumbnails(thumbnail)

        #####################
        # check segmentation
        #####################
        if not segment_result is None:
            # /!\
                # If the normalized data is 4D, then do QA on it's mean (along)
            # (the time axis) instead; if we did well, then the mean should
            # align with the template gm, wm, csf compartments pretty well,
            # and with the MNI template too; else, we've got a failed
            # normalization.
            normalized_img = ni.load(
                check_preprocessing.do_3Dto4D_merge(normalized_files))
            if len(normalized_img.shape) == 4:
                mean_normalized_img = ni.Nifti1Image(
                normalized_img.get_data().mean(-1),
                normalized_img.get_affine())
                if type(normalized_files) is str:
                    tmp = os.path.dirname(normalized_files)
                else:
                    tmp = os.path.dirname(normalized_files[0])
                mean_normalized_file = os.path.join(
                    tmp,
                    'mean%s.nii' % brain)
                ni.save(mean_normalized_img, mean_normalized_file)
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
                title=("Template GM, WM, and CSF contours on "
                       "subject's %s") % brain)

            # create thumbnail
            if results_gallery:
                thumbnail = reporter.Thumbnail()
                thumbnail.a = reporter.a(
                    href=os.path.basename(template_compartments_contours))
                thumbnail.img = reporter.img(
                    src=os.path.basename(template_compartments_contours),
                    height="500px")
                thumbnail.description = (
                    "Normalization (<a href=%s>see "
                    "execution log</a>)") % (nipype_html_report_filename)

                results_gallery.commit_thumbnails(thumbnail)

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
                title=("Subject's GM, WM, and CSF contours on "
                       "subject's %s") % brain)

            # create thumbnail
            if results_gallery:
                thumbnail = reporter.Thumbnail()
                thumbnail.a = reporter.a(
                    href=os.path.basename(subject_compartments_contours))
                thumbnail.img = reporter.img(
                    src=os.path.basename(subject_compartments_contours),
                    height="500px")
                thumbnail.description = \
                    "Normalization (<a href=%s>see execution log</a>)" \
                    % nipype_html_report_filename

                results_gallery.commit_thumbnails(thumbnail)

            output['axials'] = {}
            output['axial'] = template_compartments_contours_axial

    # collect ouput
    output['result'] = norm_result

    return output


def do_subject_preproc(
    subject_data,
    delete_orientation=False,
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

    # create subject_data.output_dir if dir doesn't exist
    if not os.path.exists(subject_data.output_dir):
        os.makedirs(subject_data.output_dir)

    # sanity
    if delete_orientation:
        subject_data.delete_orientation()

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
            results_gallery=results_gallery,
            in_files=subject_data.func,
            register_to_mean=True,
            jobtype='estwrite',
            )

        # collect output
        realign_result = realign_output['result']
        subject_data.func = realign_result.outputs.realigned_files
        mean_func = realign_result.outputs.mean_image

        # generate report stub
        if do_report:
            final_thumbnail.img.src = realign_output['rp_plot']
    else:
        # manually compute mean (along time axis) of fMRI images
        func_images = ni.load(subject_data.func)
        mean_func_image = ni.Nifti1Image(
            func_images.get_data().mean(-1), func_images.get_affine())
        mean_func = os.path.join(
            os.path.dirname(subject_data.func),
            'mean' + os.path.basename(subject_data.func))
        ni.save(mean_func_image, mean_func)

    ################################################################
    # co-registration of structural (anatomical) against functional
    ################################################################
    if do_coreg:
        # specify input files for coregistration
        coreg_target = mean_func
        coreg_jobtype = 'estimate'
        if subject_data.anat is None:
            coreg_source = EPI_TEMPLATE
            coreg_jobtype = 'estwrite'
            do_segment = False
        else:
            coreg_source = subject_data.anat

        # run coreg proper
        coreg_output = do_subject_coreg(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            do_report=do_report,
            results_gallery=results_gallery,
            comments="anat -> epi",
            target=coreg_target,
            source=coreg_source,
            jobtype=coreg_jobtype,
            )

        # collect results
        coreg_result = coreg_output['result']

        # rest anat to coregistered version thereof
        subject_data.anat = coreg_result.outputs.coregistered_source

        # generate report stub
        if do_report:
            final_thumbnail.img.src = coreg_output['axial_outline']

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
            )

        segment_result = segment_output['result']

        ##############################################################
        # indirect normalization: warp fMRI images int into MNI space
        # using the deformations learned by segmentation
        ##############################################################
        norm_parameter_file = segment_result.outputs.transformation_mat
        norm_apply_to_files = subject_data.func

        norm_output = do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            segment_result=segment_result,
            do_report=True,
            results_gallery=results_gallery,
            brain='epi',
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=get_vox_dims(norm_apply_to_files),
            write_interp=1,
            jobtype='write',
            )

        norm_result = norm_output["result"]
        subject_data.func = norm_result.outputs.normalized_files

        if do_report:
            final_thumbnail.img.src = \
                norm_output['axial']

        #############################################################
        # indirect normalization: warp anat image int into MNI space
        # using the deformations learned by segmentation
        #############################################################
        import pylab as pl

        norm_parameter_file = segment_result.outputs.transformation_mat
        norm_apply_to_files = subject_data.anat

        norm_output = do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            segment_result=segment_result,
            brain="anat",
            cmap=pl.cm.gray,
            do_report=do_report,
            results_gallery=results_gallery,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=get_vox_dims(norm_apply_to_files),
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write')
    else:
        ############################################
        # learn T1 deformation without segmentation
        ############################################
        norm_output = do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            source=subject_data.anat,
            template=T1_TEMPLATE,
            do_report=False)

        norm_result = norm_output['result']

        ####################################################
        # Warp EPI into MNI space using learned deformation
        ####################################################
        norm_parameter_file = norm_result.outputs.normalization_parameters
        norm_apply_to_files = subject_data.func

        norm_output = do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            brain="epi",
            do_report=do_report,
            results_gallery=results_gallery,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=get_vox_dims(norm_apply_to_files),
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write')

        norm_result = norm_output["result"]
        subject_data.func = norm_result.outputs.normalized_files

        #####################################################
        # Warp anat into MNI space using learned deformation
        #####################################################
        import pylab as pl

        norm_parameter_file = norm_result.outputs.normalization_parameters
        norm_apply_to_files = subject_data.anat

        norm_output = do_subject_normalize(
            subject_data.output_dir,
            subject_id=subject_data.subject_id,
            brain="anat",
            cmap=pl.cm.gray,
            do_report=do_report,
            results_gallery=results_gallery,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            write_bounding_box=[[-78, -112, -50], [78, 76, 85]],
            write_voxel_sizes=get_vox_dims(norm_apply_to_files),
            write_wrap=[0, 0, 0],
            write_interp=1,
            jobtype='write')

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

            corrected_FMRIs = list([subject_data.func])

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
                     delete_orientation=False,
                     do_report=True,
                     do_export_report=False,
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

    kwargs = {'delete_orientation': delete_orientation,
              'do_realign': do_realign, 'do_coreg': do_coreg,
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

        preproc_undergone = "<ul>"

        if do_bet:
            preproc_undergone += (
                "<li>"
                "Brain extraction has been applied to strip-off the skull"
                " and other non-brain components from the subject's "
                "anatomical image. This prevents later registration problems "
                "like the skull been (mis-)aligned unto the cortical surface, "
                "etc.</li>")
        if do_realign:
            preproc_undergone += (
                "<li>"
                "Motion correction has been done so as to detect artefacts"
                " due to the subject's head motion during the acquisition, "
                "after which the images have been resliced.</li>")
        if do_coreg:
            preproc_undergone += (
                "<li>"
                "The subject's anatomical image has been coregistered "
                "against their fMRI images (precisely, to the mean thereof). "
                "Coregistration is important as it allows deformations of the "
                "anatomy (learned by registration against other images, "
                "templates for example) to be directly applicable to the fMRI."
                "</li>")
        if do_segment:
            preproc_undergone += (
                "<li>"
                "Tissue Segmentation has been employed to segment the "
                "anatomical image into GM, WM, and CSF compartments by using "
                "TPMs (Tissue Probability Maps) as priors.</li>")

            preproc_undergone += (
                "<li>"
                "The segmented anatomical image has been warped "
                "into the MNI template space by applying the deformations "
                "learned during segmentation. The same deformations have been"
                " applied to the fMRI images. This procedure is referred to "
                "as <i>indirect Normalization</i> in SPM jargon.</li>")

        preproc_undergone += "</ul>"

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

        kwargs['parent_results_gallery'] = results_gallery
        kwargs['main_page'] = "../../%s" % os.path.basename(report_filename)

    # preproc subjects
    joblib.Parallel(n_jobs=N_JOBS, verbose=100)(joblib.delayed(
            do_subject_preproc)(
            subject_data, **kwargs) for subject_data in subjects)

    print "HTML report (dynamic) written to %s" % report_filename

    # export report (so it can be emailed, for example)
    if do_report:
        if do_export_report:
            reporter.export_report(os.path.dirname(report_filename))
