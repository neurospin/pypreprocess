"""
Author: Bertrand Thirion, Alexandre Abraham, DOHMATOB Elvis Dopgima

"""

import os
import subprocess
import nipype.interfaces.fsl as fsl
from nipype.caching import Memory as NipypeMemory
from joblib import Memory  as JoblibMemory

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
FSL_T1_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"


def _get_file_ext(filename):
    parts = filename.split('.')

    return parts[0], ".".join(parts[1:])


def _get_output_filename(input_filename, output_dir, output_prefix='',
                         ext=None):
    if isinstance(input_filename, str):
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
    print(cmdline)
    print(subprocess.check_output(cmdline))

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
    if not isinstance(output['func'], str):
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
