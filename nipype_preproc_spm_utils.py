"""
:Module: nipype_preproc_spm_utils
:Synopsis: routine functions for SPM preprocessing business
:Author: dohmatob elvis dopgima

XXX TODO: document the code!
XXX TODO: re-factor the code!
"""

# standard imports
import os
import shutil

# for handling nifti
import nibabel

# imports for caching (yeah, we aint got time to loose!)
from nipype.caching import Memory as nipypeMemory
from joblib import Memory as joblibMemory

# spm and matlab imports
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab

# parallelism imports
from joblib import Parallel, delayed
from multiprocessing import cpu_count

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
MATLAB_SPM_DIR = '/i2bm/local/spm8'
if 'MATLAB_SPM_DIR' in os.environ:
    MATLAB_SPM_DIR = os.environ['MATLAB_SPM_DIR']
assert os.path.exists(MATLAB_SPM_DIR), \
    "nipype_preproc_smp_utils: MATLAB_SPM_DIR: %s,\
 doesn't exist; you need to export MATLAB_SPM_DIR" % MATLAB_SPM_DIR
matlab.MatlabCommand.set_default_paths(MATLAB_SPM_DIR)

# set job count
N_JOBS = -1
if 'N_JOBS' in os.environ:
    N_JOBS = int(os.environ['N_JOBS'])

# set templates
T1_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'templates/T1.nii')
GM_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/grey.nii')
WM_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/white.nii')
CSF_TEMPLATE = os.path.join(MATLAB_SPM_DIR, 'tpm/csf.nii')


def do_subject_realign(output_dir,
                       subject_id=None,
                       do_report=True,
                       **kwargs):
    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = nipypeMemory(base_dir=cache_dir)

    # run workflow
    realign = mem.cache(spm.Realign)
    realign_result = realign(**kwargs)

    # generate gallery for HTML report
    if do_report:
        import check_preprocessing
        import tempita
        import reporter

        rp = realign_result.outputs.realignment_parameters

        nipype_report_filename = os.path.join(
            os.path.dirname(rp),
            "_report/report.rst")

        rp_plot = check_preprocessing.plot_spm_motion_parameters(
        rp,
        subject_id=subject_id,
        title="Plot of motion parameters before realignment")

        nipype_html_report_filename = os.path.join(
            output_dir, os.path.basename(nipype_report_filename) + '.html')
        nipype_report = reporter.nipype2htmlreport(nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        gallery = tempita.bunch(
            title="Motion Correction",
            thumbnails=[tempita.bunch(
                    title=None,
                    image=rp_plot)],
            nb_thumbnails_per_row=1,
            exec_log=nipype_html_report_filename)

        output['gallery'] = gallery

    # collect ouput
    output['result'] = realign_result
    output['rp_plot'] = rp_plot

    return output


def do_subject_coreg(output_dir,
                     subject_id=None,
                     do_report=True,
                     cmap=None,
                     **kwargs):
    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = nipypeMemory(base_dir=cache_dir)

    # run workflow
    coreg = mem.cache(spm.Coregister)
    coreg_result = coreg(**kwargs)

    # generate gallery for HTML report
    if do_report:
        import check_preprocessing
        import tempita
        import reporter

        target = os.path.basename(kwargs['target'])
        source = os.path.basename(kwargs['source'])

        # prepare for smart caching
        qa_cache_dir = os.path.join(output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblibMemory(cachedir=qa_cache_dir, verbose=5)

        overlap = os.path.join(
            output_dir,
            "%s_on_%s_overlap.png" % (target, source))
        qa_mem.cache(check_preprocessing.plot_registration)(
            kwargs['target'],
            coreg_result.outputs.coregistered_source,
            output_filename=overlap,
            cmap=cmap,
            title="Overlap of %s on %s" % (target, source))

        overlap_summary = os.path.join(
            output_dir,
            "%s_on_%s_overlap_summary.png" % (target, source))
        qa_mem.cache(check_preprocessing.plot_registration)(
            kwargs['target'],
            coreg_result.outputs.coregistered_source,
            output_filename=overlap_summary,
            slicer='z',
            cmap=cmap,
            title="%s: coreg" % subject_id)

        # convert nipype log to HTML
        nipype_report_filename = os.path.join(
            os.path.dirname(coreg_result.outputs.coregistered_source),
            "_report/report.rst")
        nipype_html_report_filename = os.path.join(
            output_dir,
            os.path.basename(nipype_report_filename) + '.html')
        nipype_report = reporter.nipype2htmlreport(
            nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        gallery = tempita.bunch(
            title="Coregistration",
            thumbnails=[tempita.bunch(
                    title=None,
                    image=overlap)],
            nb_thumbnails_per_row=1,
            exec_log=nipype_html_report_filename)

        output['gallery'] = gallery
        output['summary'] = overlap_summary

    # collect ouput
    output['result'] = coreg_result

    return output


def do_subject_segment(output_dir,
                       subject_id=None,
                       do_report=True,
                       **kwargs):
    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = nipypeMemory(base_dir=cache_dir)

    # run workflow
    segment = mem.cache(spm.Segment)
    segment_result = segment(**kwargs)

    # generate gallery for HTML report
    if do_report:
        pass

    # collect ouput
    output['result'] = segment_result

    return output


def do_subject_normalize(output_dir,
                         subject_id=None,
                         do_report=True,
                         preproc_stream='segmentation',
                         cmap=None,
                         segment_result=None,
                         data_name="brain",
                         **kwargs):
    output = {}

    # prepare for smart caching
    cache_dir = os.path.join(output_dir, 'cache_dir')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = nipypeMemory(base_dir=cache_dir)

    # run workflow
    normalize = mem.cache(spm.Normalize)
    norm_result = normalize(**kwargs)

    # generate gallery for HTML report
    if do_report and preproc_stream == "segmentation":
        import check_preprocessing
        import tempita
        import reporter

        # prepare for smart caching
        qa_cache_dir = os.path.join(output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblibMemory(cachedir=qa_cache_dir, verbose=5)

        gallery_thumbnails = []
        gallery = tempita.bunch(nb_thumbnails_per_row=1)

        # plot contours of template TPMs on subject's brain
        template_tpms_contours = os.path.join(
            output_dir,
            "template_tmps_contours_on_%s.png" % data_name)
        template_tpms_contours_summary = os.path.join(
            output_dir,
            "template_tpms_contours_on_%s_summary.png" % data_name)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            norm_result.outputs.normalized_files,
            GM_TEMPLATE,
            WM_TEMPLATE,
            CSF_TEMPLATE,
            output_filename=template_tpms_contours_summary,
            slicer='z',
            cmap=cmap,
            title="%s: template TPMs" % subject_id)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            norm_result.outputs.normalized_files,
            GM_TEMPLATE,
            WM_TEMPLATE,
            CSF_TEMPLATE,
            output_filename=template_tpms_contours,
            cmap=cmap,
            title="Template GM, WM, and CSF contours on subject's %s" % \
                data_name)

        gallery_thumbail = tempita.bunch(title=None,
                                         image=template_tpms_contours)
        gallery_thumbnails.append(gallery_thumbail)

        # plot contours of subject's TPMs on subjects brain
        subject_tpms_contours = os.path.join(
            output_dir,
            "subject_tmps_contours_on_subject_%s.png" % data_name)
        subject_tpms_contours_summary = os.path.join(
            output_dir,
            "subject_tmps_contours_on_subject_%s_summary.png" % data_name)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            norm_result.outputs.normalized_files,
            segment_result.outputs.modulated_gm_image,
            segment_result.outputs.modulated_wm_image,
            segment_result.outputs.modulated_csf_image,
            output_filename=subject_tpms_contours_summary,
            slicer='z',
            cmap=cmap,
            title="%s: subject TPMs" % subject_id)

        qa_mem.cache(check_preprocessing.plot_segmentation)(
            norm_result.outputs.normalized_files,
            segment_result.outputs.modulated_gm_image,
            segment_result.outputs.modulated_wm_image,
            segment_result.outputs.modulated_csf_image,
            output_filename=subject_tpms_contours,
            title="Subject's GM, WM, and CSF contours on subject's %s" % \
                data_name)

        gallery_thumbail = tempita.bunch(title=None,
                                         image=subject_tpms_contours)
        gallery_thumbnails.append(gallery_thumbail)

        # convert nipype log to HTML
        nipype_report_filename = os.path.join(
            os.path.dirname(norm_result.outputs.normalized_files),
            "_report/report.rst")
        nipype_html_report_filename = os.path.join(
            output_dir,
            os.path.basename(nipype_report_filename) + '.html')
        nipype_report = reporter.nipype2htmlreport(
            nipype_report_filename)
        open(nipype_html_report_filename, 'w').write(str(nipype_report))

        gallery.thumbnails = gallery_thumbnails
        gallery.exec_log = nipype_html_report_filename
        output['gallery'] = gallery

        output['axials'] = {}
        output['axials']['template_tpms_contours'] = \
            template_tpms_contours_summary
        output['axials']['subject_tpms_contours'] = \
            subject_tpms_contours_summary

    # collect ouput
    output['result'] = norm_result

    return output


def do_subject_preproc(
    subject_data,
    do_report=True,
    do_realign=True,
    do_coreg=True,
    do_segment=True,
    **preproc_kwargs):
    """
    Function preprocessing data for a single subject.

    """

    # unpack (and sanitize) input
    output_dir = subject_data['output_dir']
    anat_image = subject_data['anat']
    fmri_images = subject_data['func']
    if not 'subject_id' in subject_data:
        subject_id = None
    else:
        subject_id = subject_data['subject_id']
    if not 'session_id' in subject_data:
        session_id = None
    else:
        session_id = subject_data['session_id']

    if do_report:
        import check_preprocessing
        import tempita
        import reporter
        from nipy.labs import viz

        report_filename = os.path.join(output_dir, "_report.html")
        galleries = []
        output = {}
        output["plots"] = {}
        final_thumbnail = tempita.bunch(title=subject_id)

    #  motion correction
    if do_realign:
        realign_output = do_subject_realign(
            output_dir,
            subject_id=subject_id,
            do_report=do_report,
            in_files=fmri_images,
            register_to_mean=True,
            jobtype='estwrite')

        realign_result = realign_output['result']

        mean_image = realign_result.outputs.mean_image
        final_func = realign_result.outputs.realigned_files

        if do_report:
            galleries.append(realign_output['gallery'])
            final_thumbnail.image = realign_output['rp_plot']
    else:
        img = nibabel.load(fmri_images)
        mean = nibabel.Nifti1Image(
            img.get_data().mean(-1), img.get_affine())
        mean_image = os.path.join(os.path.dirname(fmri_images),
                                  'mean' + os.path.basename(fmri_images))
        nibabel.save(mean, mean_image)

    # co-registration of functional against structural (anatomical)
    if do_coreg:
        coreg_target = anat_image
        coreg_source = mean_image
        coreg_apply_to_files = final_func
        coreg_output = do_subject_coreg(output_dir,
                                        subject_id=subject_id,
                                        do_report=do_report,
                                        target=coreg_target,
                                        source=coreg_source,
                                        apply_to_files=coreg_apply_to_files,
                                        jobtype='estimate',  # XXX why ?
                                        )

        coreg_result = coreg_output['result']

        final_func = coreg_result.outputs.coregistered_files

        if do_report:
            gallery = coreg_output['gallery']
            gallery.title = 'Coregistration (func -> anat)'
            galleries.append(coreg_output['gallery'])
            final_thumbnail.image = coreg_output['summary']

    # segment anat
    if do_segment:
        segment_data = anat_image
        segment_output = do_subject_segment(
            output_dir,
            subject_id=subject_id,
            do_report=False,  # XXX why ?
            data=segment_data,
            gm_output_type=[True, True, True],
            wm_output_type=[True, True, True],
            csf_output_type=[True, True, True],
            tissue_prob_maps=[GM_TEMPLATE,
                              WM_TEMPLATE, CSF_TEMPLATE])

        segment_result = segment_output['result']

        # if do_report:
        #     galleries.append(segment_output['gallery'])

        # normalize anat based on the learned segmentation
        norm_parameter_file = segment_result.outputs.transformation_mat
        norm_apply_to_files = anat_image

        norm_output = do_subject_normalize(output_dir,
                                           subject_id=subject_id,
                                           segment_result=segment_result,
                                           data_name="anat",
                                           cmap=viz.cm.cold_hot,
                                           do_report=do_report,
                                           parameter_file=norm_parameter_file,
                                           apply_to_files=norm_apply_to_files,
                                           jobtype='write')

        norm_result = norm_output['result']

        if do_report:
            gallery = norm_output['gallery']
            gallery.title = 'Segmentation of anatomical image'
            galleries.append(gallery)

        # normalize mean func based on the learned segmentation
        norm_parameter_file = segment_result.outputs.transformation_mat
        if do_coreg:
            norm_apply_to_files = coreg_result.outputs.coregistered_source
        else:
            norm_apply_to_files = mean_image

        norm_output = do_subject_normalize(
            output_dir,
            subject_id=subject_id,
            segment_result=segment_result,
            data_name="epi",
            do_report=do_report,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            jobtype='write',
                                               )
        norm_result = norm_output['result']

        if do_report:
            gallery = norm_output['gallery']
            gallery.title = 'Segmentation of mean functional image'
            galleries.append(gallery)
            final_thumbnail.image =\
                norm_output['axials']['subject_tpms_contours']

        # normalize func images based on the learned segmentation
        norm_parameter_file = segment_result.outputs.transformation_mat
        if do_coreg:
            norm_apply_to_files = coreg_result.outputs.coregistered_files
        else:
            norm_apply_to_files = fmri_images

        norm_output = do_subject_normalize(
            output_dir,
            subject_id=subject_id,
            do_report=False,
            parameter_file=norm_parameter_file,
            apply_to_files=norm_apply_to_files,
            jobtype='write',
            )

        norm_result = norm_output['result']
        final_func = norm_result.outputs.normalized_files

    # generate html report (for QA)
    if do_report:
        blablabla = "Generating QA reports for subject %s .." % subject_id
        dadada = "+" * len(blablabla)
        print "\r\n%s\r\n%s\r\n%s\r\n" % (dadada, blablabla, dadada)

        qa_cache_dir = os.path.join(output_dir, "QA")
        if not os.path.exists(qa_cache_dir):
            os.makedirs(qa_cache_dir)
        qa_mem = joblibMemory(cachedir=qa_cache_dir, verbose=5)

        cv_tc_plot_after = os.path.join(output_dir, "cv_tc_after.png")

        corrected_FMRIs = [final_func]
        qa_mem.cache(check_preprocessing.plot_cv_tc)(
            corrected_FMRIs, [session_id], subject_id,
            output_dir,
            cv_tc_plot_outfile=cv_tc_plot_after,
            plot_diff=True,
            title="subject %s after preproc " % subject_id)

        gallery = tempita.bunch(
            title="Coefficient of Variation",
            thumbnails=[tempita.bunch(
                    title=None,
                    image=cv_tc_plot_after)],
            nb_thumbnails_per_row=1)
        galleries.append(gallery)

        final_thumbnail.tooltip_image = cv_tc_plot_after

        report = reporter.SUBJECT_PREPROC_REPORT_HTML_TEMPLATE.substitute(
            subject_id=subject_id, galleries=galleries)

        print ">" * 80 + "BEGIN HTML"
        print report
        print "<" * 80 + "END HTML"

        with open(report_filename, 'w') as fd:
            fd.write(str(report))
            fd.close()

        final_thumbnail.redirect_url = report_filename
        output['final_thumbnail'] = final_thumbnail

        return subject_id, session_id, output


def do_group_preproc(subjects,
                     do_report=True,
                     dataset_description=None,
                     report_filename=None,
                     do_realign=True,
                     do_coreg=True,
                     do_segment=True):

    kwargs = {'do_realign': do_realign, 'do_coreg': do_coreg,
                'do_segment': do_segment}

    # preproc subjects
    results = Parallel(n_jobs=1)(delayed(do_subject_preproc)(
            subject_data, **kwargs) for subject_data in subjects)

    # generate html report (for QA) as desired
    if do_report:
        import tempita
        import reporter
        blablabla = "Generating QA report for %d subjects .." % len(results)
        dadada = "+" * len(blablabla)
        print "\r\n%s\r\n%s\r\n%s\r\n" % (dadada, blablabla, dadada)

        tmpl = reporter.DATASET_PREPROC_REPORT_HTML_TEMPLATE

        gallery = tempita.bunch()

        if do_segment:
            gallery.nb_thumbnails_per_row = 4
        elif do_coreg:
            gallery.nb_thumbnails_per_row = 4
        elif do_realign:
            gallery.nb_thumbnails_per_row = 2

        gallery_thumbnails = []

        # populate gallery
        for subject_id, session_id, output in results:
            gallery_thumbnails.append(output['final_thumbnail'])

        gallery.thumbnails = gallery_thumbnails

        # XXX the following info is fake; compute it from function args
        preproc_undergone = """\
<p>All preprocessing has been done using nipype's interface to the SPM8 \
package.</p>
<p>Only intra-subject preprocessing has been carried out. For each \
subject:<br>
1. motion correction has been done so as to detect artefacts due to the \
subject's head motion during the acquisition, after which the images \
have been resliced;<br>
2. the fMRI images (a 4D time-series made of 3D volumes aquired every TR \
seconds) have been coregistered against the subject's anatomical. At the \
end of this stage, the fMRI images have gain some anatomical detail, useful \
for warping the fMRI into some standard space later on;<br>
3. the subject's anatomical has been segmented into GM, WM, and CSF tissue \
probabitility maps (TPMs);</p>
"""
        report = tmpl.substitute(dataset_description=dataset_description,
                                 preproc_undergone=preproc_undergone,
                                 gallery=gallery)

        print ">" * 80 + "BEGIN HTML"
        print report
        print "<" * 80 + "END HTML\r\n"

        if not report_filename is None:
            with open(report_filename, 'w') as fd:
                fd.write(str(report))
                fd.close()
                print "HTML report written to %s" % report_filename

        print "\r\nDone."
