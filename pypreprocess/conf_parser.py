import os
import glob
from configobj import ConfigObj
import numpy as np
from subject_data import SubjectData


def _parse_job(jobfile):
    assert os.path.isfile(jobfile)

    def sanitize(section, key):
        val = section[key]

        if key == "slice_order":
            if isinstance(val, basestring):
                return

        if isinstance(val, basestring):
            if val.lower() in ["true", "yes"]:
                val = True
            elif val.lower() in ["false", "no"]:
                val = False
            elif key == "slice_order":
                val = val.lower()
            elif val.lower() in ["none", "auto"]:
                val = None

        if key in ["TR", "nslices", "refslice"]:
            if not val is None:
                val = eval(val)

        if key in ["fwhm", "anat_voxel_sizes", "func_voxel_sizes",
                   "slice_order"]:
            dtype = np.int if key == "slice_order" else np.float
            val = ",".join(val).replace("[", "")
            val = val.replace("]", "")
            val = list(np.fromstring(val, sep=",", dtype=dtype))
            if len(val) == 1:
                val = val[0]

        section[key] = val

    cobj = ConfigObj(jobfile)
    cobj.walk(sanitize, call_on_sections=True)

    return cobj


def _generate_preproc_pipeline(jobfile):

    # read config file
    jobfile = os.path.abspath(jobfile)
    preproc_options = _parse_job(jobfile)

    # generate subject conf
    subjects = []
    data_dir = os.path.abspath(os.path.dirname(jobfile))
    old_cwd = os.getcwd()
    os.chdir(data_dir)

    output_dir = preproc_options["Output"]["output_dir"]
    if output_dir.startswith("./"):
        output_dir = output_dir[2:]
    elif output_dir.startswith("."):
        output_dir = output_dir[1:]
    output_dir = os.path.abspath(output_dir)
    for subject_data_dir in glob.glob(os.path.join(
            data_dir, preproc_options['Input'][
                "subject_dir_prefix"] + "*")):
        subject_id = os.path.basename(subject_data_dir)

        # grab functional data
        sess_dir_wildcard = preproc_options['Input']["session_dir_wildcard"]
        if sess_dir_wildcard in [".", None]:
            sess_dir_wildcard = ""
        func = sorted(glob.glob(os.path.join(
                    subject_data_dir, sess_dir_wildcard,
                    preproc_options['Input']["func_basename_wildcard"])))

        # grab anat
        anat_dir = preproc_options["Input"]["anat_dir"]
        if anat_dir in [".", None]:
            anat_dir = ""
        anat = glob.glob(os.path.join(subject_data_dir,
                                      anat_dir,
                                      preproc_options['Input']["anat_basename"]
                                      ))
        assert len(anat) == 1
        anat = anat[0]

        # make subject data
        subject_data = SubjectData(func=func, anat=anat,
                                   output_dir=os.path.join(output_dir,
                                                           subject_id))

        subjects.append(subject_data)

    preproc_params = {"do_report": preproc_options["Output"]["report"],
                      "output_dir": output_dir,
                      "dataset_id": preproc_options['Input']["dataset_id"]}
    if preproc_params["dataset_id"] is None:
        preproc_params["dataset_id"] = data_dir

    # configure slice-timing correction node
    preproc_params["do_slice_timing"] = not preproc_options[
        "SliceTiming"]["disable"]
    if preproc_params['do_slice_timing']:
        stc_options = preproc_options['SliceTiming']
        preproc_params["do_slice_timing"] = True
        preproc_params.update(dict((k, stc_options[k])
                                   for k in ["TR", "TA", "slice_order",
                                             "interleaved"]))

    # configure motion correction node
    preproc_params["do_realign"] = not preproc_options[
        "Realign"]["disable"]
    if not preproc_params["do_realign"]:
        realign_options = preproc_options["Realign"]
        preproc_params['realign_reslice'] = realign_options["reslice"]
        preproc_params['register_to_mean'] = realign_options[
            "register_to_mean"]

    # configure coregistration node
    preproc_params["do_coreg"] = not preproc_options[
        "Coregister"]["disable"]
    if preproc_params["do_coreg"]:
        coreg_options = preproc_options["Coregister"]
        preproc_params['coreg_reslice'] = coreg_options["reslice"]
        preproc_params['coreg_anat_to_func'] = not coreg_options[
            "func_to_anat"]

    # configure tissue segmentation node
    preproc_params["do_segment"] = not preproc_options[
        "Segment"]["disable"]

    # configure normalization node
    preproc_params["do_normalize"] = not preproc_options[
        "Normalize"]["disable"]
    if preproc_params["do_normalize"]:
        normalize_options = preproc_options["Normalize"]
        preproc_params['func_write_voxel_sizes'] = normalize_options[
            "func_voxel_sizes"]
        preproc_params['anat_write_voxel_sizes'] = normalize_options[
            "anat_voxel_sizes"]
        preproc_params['do_dartel'] = normalize_options["dartel"]

    # configure smoothing node
    if not preproc_options["Smooth"]["disable"]:
        preproc_params["fwhm"] = preproc_options["Smooth"]["fwhm"]

    os.chdir(old_cwd)

    return subjects, preproc_params

if __name__ == '__main__':
    from pypreprocess.reporting.base_reporter import dict_to_html_ul
    print dict_to_html_ul(_parse_job("job.conf"))
