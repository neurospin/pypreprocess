import os
from nilearn.datasets import fetch_haxby
from pypreprocess.nipype_preproc_spm_utils import (do_subjects_preproc,
                                                    SubjectData)

def test_reporting():
    # fetch HAXBY dataset
    haxby_data = fetch_haxby(subjects=1)

    # set output dir
    output_dir = os.path.join(os.path.dirname(haxby_data.mask), "haxby_runs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # instantiate subject_data object
    subject_data = SubjectData()
    subject_data.subject_id = 'subj1'
    subject_data.session_id = "haxby2001"

    # set func
    subject_data.func = haxby_data.func[0]
    subject_data.anat = haxby_data.anat[0]
    subject_data.output_dir = os.path.join(output_dir,subject_data.subject_id)

    # do preprocessing proper
    result = do_subjects_preproc([subject_data],output_dir=output_dir,
                                    dataset_id="HAXBY 2001",report=True)
