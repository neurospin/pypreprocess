import sys
import os
sys.path.append("pypreprocess/")
from spm_loader.spm import load_intra
from nipype_preproc_spm_utils import _do_subject_realign
from spm_loader.spm import load_intra

location = ("/neurospin/unicog/protocols/IRMf/Compression_"
            "Vagharchakian_new_2009/Subjects/MS080150/fMRI/"
            "acquisition1/analysis/model7_HRF_comp_FINAL/SPM.mat")
dotmat = load_intra(location, squeeze_me=True, struct_as_record=False)
sessions = set([os.path.basename(x).split('_')[5] for x in dotmat['raw_data']])
files = [sorted([x for x in dotmat['raw_data'] if s in x])
         for s in sessions]
_tmp = _do_subject_realign('/tmp/', sessions=sessions,
                        do_report=False,
                        in_files=files, register_to_mean=True)
