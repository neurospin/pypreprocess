"""
Author: Bertrand Thirion, Alexandre Abraham, DOHMATOB Elvis Dopgima

"""

import os
import commands
import nipype.interfaces.fsl as fsl
from nipype.caching import Memory as NipypeMemory
from joblib import Parallel, delayed
from sklearn.externals.joblib import Memory as JoblibMemory

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
FSL_T1_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"


def _get_file_ext(filename):
    parts = filename.split('.')

    return parts[0], ".".join(parts[1:])


def _get_output_filename(input_filename, output_dir, output_prefix='',
                         ext=None):
    if isinstance(input_filename, basestring):
        if not ext is None:
            ext = "." + ext if not ext.startswith('.') else ext
            input_filename = _get_file_ext(input_filename)[0] + ext

        return os.path.join(output_dir,
                            output_prefix + os.path.basename(input_filename))
    else:
        return [_get_output_filename(x, output_dir,
                                     output_prefix=output_prefix)
                for x in input_filename]


def do_fsl_merge(in_files, output_dir, output_prefix='merged_',
                 cmd_prefix="fsl5.0-"
                 ):
    output_filename = _get_output_filename(in_files[0], output_dir,
                                           output_prefix=output_prefix,
                                           ext='.nii.gz')

    cmdline = "%sfslmerge -t %s %s" % (cmd_prefix, output_filename,
                                       " ".join(in_files))
    print cmdline
    print commands.getoutput(cmdline)

    return output_filename


def do_subject_preproc(subject_id,
                       output_dir,
                       func,
                       anat,
                       do_bet=True,
                       do_mc=True,
                       do_coreg=True,
                       do_normalize=True,
                       cmd_prefix="fsl5.0-",
                       **kwargs
                       ):
    """
    Preprocesses subject data using FSL.

    Parameters
    ----------

    """

    output = {'func': func,
              'anat': anat
              }

    # output dir
    subject_output_dir = os.path.join(output_dir, subject_id)
    if not os.path.exists(subject_output_dir):
        os.makedirs(subject_output_dir)

    # prepare for smart-caching
    cache_dir = os.path.join(output_dir, "cache_dir")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    nipype_mem = NipypeMemory(base_dir=cache_dir)
    joblib_mem = JoblibMemory(cache_dir, verbose=100)

    # sanitize input files
    if not isinstance(output['func'], basestring):
        output['func'] = joblib_mem.cache(do_fsl_merge)(
            func, subject_output_dir, output_prefix='Merged',
            cmd_prefix=cmd_prefix)

    ######################
    #  Brain Extraction
    ######################
    if do_bet:
        if not fsl.BET._cmd.startswith("fsl"):
            fsl.BET._cmd = cmd_prefix + fsl.BET._cmd

        bet = nipype_mem.cache(fsl.BET)
        bet_results = bet(in_file=output['anat'],
                          )

        output['anat'] = bet_results.outputs.out_file

    #######################
    #  Motion correction
    #######################
    if do_mc:
        if not fsl.MCFLIRT._cmd.startswith("fsl"):
            fsl.MCFLIRT._cmd = cmd_prefix + fsl.MCFLIRT._cmd

        mcflirt = nipype_mem.cache(fsl.MCFLIRT)
        mcflirt_results = mcflirt(in_file=output['func'],
                                  cost='mutualinfo',
                                  save_mats=True,  # save mc matrices
                                  save_plots=True  # save mc params
                                  )

        output['motion_parameters'] = mcflirt_results.outputs.par_file
        output['motion_matrices'] = mcflirt_results.outputs.mat_file
        output['func'] = mcflirt_results.outputs.out_file

    ###################
    # Coregistration
    ###################
    if do_coreg:
        if not fsl.FLIRT._cmd.startswith("fsl"):
            fsl.FLIRT._cmd = cmd_prefix + fsl.FLIRT._cmd

        flirt1 = nipype_mem.cache(fsl.FLIRT)
        flirt1_results = flirt1(in_file=output['func'],
                                reference=output['anat']
                                )

        if not do_normalize:
            output['func'] = flirt1_results.outputs.out_file

    ##########################
    # Spatial normalization
    ##########################
    if do_normalize:
        if not fsl.FLIRT._cmd.startswith("fsl"):
            fsl.FLIRT._cmd = cmd_prefix + fsl.FLIRT._cmd

        # T1 normalization
        flirt2 = nipype_mem.cache(fsl.FLIRT)
        flirt2_results = flirt2(in_file=output['anat'],
                                reference=FSL_T1_TEMPLATE)

        output['anat'] = flirt2_results.outputs.out_file

        # concatenate 'func -> anat' and 'anat -> standard space'
        # transformation matrices to obtaun 'func -> standard space'
        # transformation matrix
        if do_coreg:
            if not fsl.ConvertXFM._cmd.startswith("fsl"):
                fsl.ConvertXFM._cmd = cmd_prefix + fsl.ConvertXFM._cmd

                convertxfm = nipype_mem.cache(fsl.ConvertXFM)
                convertxfm_results = convertxfm(
                    in_file=flirt1_results.outputs.out_matrix_file,
                    in_file2=flirt2_results.outputs.out_matrix_file,
                    concat_xfm=True
                    )

        # warp func data into standard space by applying
        # 'func -> standard space' transformation matrix
        if not fsl.ApplyXfm._cmd.startswith("fsl"):
            fsl.ApplyXfm._cmd = cmd_prefix + fsl.ApplyXfm._cmd

        applyxfm = nipype_mem.cache(fsl.ApplyXfm)
        applyxfm_results = applyxfm(
            in_file=output['func'],
            in_matrix_file=convertxfm_results.outputs.out_file,
            reference=FSL_T1_TEMPLATE
            )

        output['func'] = applyxfm_results.outputs.out_file

    return output


def _do_img_topup_correction(ap_realig_img, pa_realig_img, func_imgs,
                             total_readout_times=(1., 1.), memory=None,
                             tmp_dir=None):
    """
    Compute and apply topup as implemented in FSL.

    It is crucial to provide the total_readout_times of the ap_img and pa_img
    correctly in case they are different. Otherwise a value of 1 can be taken
    as default.

    More detailed documentation can be found in the example provided in the FSL
    webpage. http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP/ExampleTopupFollowedByApplytopup.

    Parameters
    ----------
    ap_realig_img: string
        path to img using negative phase-encode blips. Anterior --> Posterior
        the image should be already realigned

    pa_realig_img: string
        path to img using positive phase-encode blips. Posterior --> Anterior
        the image should be already realigned

    func_imgs: list of string
        functional imgs path, that will be corrected with topup

    total_readout_times: tuple of floats, optional (default None)
        the first float corresponds to the total_readout_times of the ap img
        and the second float to the pa img. Its crucial to provide these if
        they are different. The times should be provided in seconds.

    memory: Nipype `Memory` object, optional (default None)
        if given, then caching will be enabled

    tmp_dir: string, optional (default None)
        temporary directory to store acquisition parameters configuration file
        for topup. Must be provided in case memory is provided.
    """
    if total_readout_times is None:
        total_readout_times = (1., 1.)  # Check FSL documentation explanation

    if memory is not None:
        if tmp_dir is None:
            raise('Temporary directory was not provided with memory object')
        # Merge AP and PA images
        merge = memory.cache(fsl.Merge)
        appa_merged = merge(in_files=[ap_realig_img, pa_realig_img],
                            dimension='t')

        # Compute topup transformation
        acq_param_file = os.path.join(tmp_dir, 'acq_param.txt')
        if not os.path.isfile(acq_param_file):
            with open(acq_param_file, 'w') as acq:
                content = '0 -1 0 {0:6.5f} \n0 1 0 {0:6.5f}'
                acq.write(content % total_readout_times)

        topup = memory.cache(fsl.TOPUP)
        correction = topup(in_file=appa_merged.outputs.merged_file,
                           encoding_file=acq_param_file,
                           output_type='NIFTI', out_base='APPA_DefMap',
                           out_field='sanitycheck_DefMap',
                           out_corrected='sanitycheck_unwarped_B0')

        # Apply topup correction to images
        fieldcoef = correction.outputs.out_fieldcoef
        movpar = correction.outputs.out_movpar
        applytopup = memory.cache(fsl.ApplyTOPUP)
        return [applytopup(in_files=img, encoding_file=acq_param_file,
                           in_index=[2], in_topup_fieldcoef=fieldcoef,
                           in_topup_movpar=movpar, output_type='NIFTI',
                           method='jac') for img in func_imgs]

    else:
        # Merge AP and PA images
        appa_merged = fsl.Merge(in_files=[ap_realig_img, pa_realig_img],
                                dimension='t').run()

        # Compute topup transformation
        acq_param_file = os.path.join('/tmp', 'pypreprocess_topup',
                                      'acq_param.txt')
        if not os.path.isfile(acq_param_file):
            with open(acq_param_file, 'w') as acq:
                content = '0 -1 0 {0:6.5f} \n0 1 0 {0:6.5f}'
                acq.write(content % total_readout_times)

        correction = fsl.TOPUP(in_file=appa_merged.outputs.merged_file,
                               encoding_file=acq_param_file,
                               output_type='NIFTI',
                               out_base='APPA_DefMap',
                               out_field='sanitycheck_DefMap',
                               out_corrected='sanitycheck_unwarped_B0').run()

        # Apply topup correction to images
        fieldcoef = correction.outputs.out_fieldcoef
        movpar = correction.outputs.out_movpar
        return [fsl.ApplyTOPUP(in_files=img, encoding_file=acq_param_file,
                               in_index=[2], in_topup_fieldcoef=fieldcoef,
                               in_topup_movpar=movpar, output_type='NIFTI',
                               method='jac').run() for img in func_imgs]


def _do_subject_topup_correction(subject_data, caching=True,
                                 hardlink_output=True):
    """
    Apply topup correction to subject functional images as implemented in FSL.

    subject_data.topup is expected for any correction to take place. It must
    contain a dictionary with imgs in subject_data.func as key and a
    tuple of the form (string, string, (float, float)) corresponding to
    (ap_img, pa_img, total_readout_times) as value. Its crucial that the ap
    and pa imgs have already been realigned.

    Parameters
    ----------
    subject_data: `SubjectData` object
        subject data whose functional images are to be corrected with topup
        (subject_data.func and subject_data.topup)

    caching: bool, optional (default True)
        if true, then caching will be enabled

    **kwargs:
       additional parameters to the back-end (SPM, FSL, python)

    Returns
    -------
    subject_data: `SubjectData` object
        preprocessed subject_data. The func and anatomical fields
        (subject_data.func and subject_data.anat) now contain the
        oregistered and anatomical images functional images of the subject
    """
    corrected_func = []
    subject_data.nipype_results['topup_correction'] = []

    imgs_to_topup = [img for img in subject_data.func if img in
                     subject_data.topup]
    for img_idx, img in enumerate(imgs_to_topup):
        ap_realig_img = subject_data.topup[img][0]
        pa_realig_img = subject_data.topup[img][1]
        total_readout_times = subject_data.topup[img][2]

        if caching:
            cache_dir = cache_dir = os.path.join(subject_data.output_dir,
                                                 'cache_dir')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            subject_data.mem = NipypeMemory(base_dir=cache_dir)
            memory = subject_data.mem
            tmp_dir = os.path.join(subject_data.tmp_output_dir,
                                   '_%d' % img_idx)
            top = _do_img_topup_correction(ap_realig_img, pa_realig_img, [img],
                                           total_readout_times, memory,
                                           tmp_dir)
        else:
            top = _do_img_topup_correction(ap_realig_img, pa_realig_img, [img],
                                           total_readout_times)
        if top is None:
            subject_data.failed = True
            return subject_data
        else:
            subject_data.nipype_results['topup_correction'].append(top)
            corrected_func.append(top.outputs.out_corrected)

    # commit output files
    if hardlink_output:
        subject_data.hardlink_output_files()
    subject_data.func = corrected_func

    return subject_data.sanitize()
