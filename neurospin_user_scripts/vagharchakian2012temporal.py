import os
import sys
import glob
import multiprocessing

warning = ("%s: THIS SCRIPT MUST BE RUN FROM ITS PARENT "
           "DIRECTORY!") % sys.argv[0]
banner = "#" * len(warning)
separator = "\r\n\t"
print separator + separator.join(["", banner, warning, banner, ""])

# spm_loader path
SPM_LOADER_DIR = os.path.join(os.path.dirname(os.path.split(
            os.path.abspath(__file__))[0]),
                              "spm_loader")
sys.path.append(SPM_LOADER_DIR)

from spm_loader.spm import load_intra as load_spm
from spm_loader.utils import fix_docs, execute_glms


study = 'vagharchakian2012temporal'
root = ('/neurospin/unicog/protocols/IRMf'
        '/Compression_Vagharchakian_new_2009/Subjects')
output_dir = '/volatile/brainpedia/protocols'

contrast_names = {
    'A100': 'auditory sentences 100% duration',
    'A100-A80': 'auditory sentences 100% - 80% duration',
    'A20': 'auditory sentences 20% duration',
    'A20-A40': 'auditory sentences 20% - 40% duration',
    'A40': 'auditory sentences 40% duration',
    'A40-A20': 'auditory sentences 40% - 20% duration',
    'A40-A60': 'auditory sentences 40% - 60% duration',
    'A60': 'auditory sentences 60% duration',
    'A60-A40': 'auditory sentences 60% - 40% duration',
    'A60-A80': 'auditory sentences 60% - 80% duration',
    'A80': 'auditory sentences 80% duration',
    'A80-A100': 'auditory sentences 80% - 100% duration',
    'A80-A60': 'auditory sentences 80% - 60% duration',
    'V100': 'visual sentences 100% duration',
    'V100-V80': 'visual sentences 100% - 80% duration',
    'V20': 'visual sentences 20% duration',
    'V20-V40': 'visual sentences 20% - 40% duration',
    'V40': 'visual sentences 40% duration',
    'V40-V20': 'visual sentences 40% - 20% duration',
    'V40-V60': 'visual sentences 40% - 60% duration',
    'V60': 'visual sentences 60% duration',
    'V60-V40': 'visual sentences 60% - 40% duration',
    'V60-V80': 'visual sentences 60% - 80% duration',
    'V80': 'visual sentences 80% duration',
    'V80-V100': 'visual sentences 80% - 100% duration',
    'V80-V60': 'visual sentences 80% - 60% duration',
    'tt Audio': 'auditory sentences all compressions',
    'tt Visuel': 'visual sentences all compressions',
    }

definitions = {
    'auditory vs visual sentences': {
        'auditory sentences all compressions': 1,
        'visual sentences all compressions': -1,
        },
    'visual vs auditory sentences': {
        'auditory sentences all compressions': -1,
        'visual sentences all compressions': 1,
        },
   # 'auditory 100 - 80 sentences': {
   #     'auditory sentences 100% duration': 1,
   #     'auditory sentences 80% duration': -1,
   #     },
   # 'auditory 80 - 100 sentences': {
   #     'auditory sentences 100% duration': -1,
   #     'auditory sentences 80% duration': 1,
   #     },
   'auditory normal speed vs bottleneck': {
       'auditory sentences 100% duration': 1,
       'auditory sentences 20% duration': -1,
       },
   'auditory bottleneck vs normal speed': {
       'auditory sentences 100% duration': -1,
       'auditory sentences 20% duration': 1,
       },
   # 'visual 100 - 80 sentences': {
   #     'visual sentences 100% duration': 1,
   #     'visual sentences 80% duration': -1,
   #     },
   # 'visual 80 - 100 sentences': {
   #     'visual sentences 100% duration': -1,
   #     'visual sentences 80% duration': 1,
   #     },
   'visual normal speed vs bottleneck': {
       'visual sentences 100% duration': 1,
       'visual sentences 20% duration': -1,
       },
   'visual bottleneck vs normal speed': {
       'visual sentences 100% duration': -1,
       'visual sentences 20% duration': 1,
       },
   'visual language bottleneck vs rest': {
       'visual sentences 20% duration': 1,
       },
   'visual language normal speed vs rest': {
       'visual sentences 100% duration': 1,
       },
   'auditory language normal speed vs rest': {
       'auditory sentences 100% duration': 1,
       },
   'auditory language bottleneck vs rest': {
       'auditory sentences 20% duration': 1,
       },
}


def get_docs(inputs=False):
    n_jobs = 24

    docs = []
    pool = multiprocessing.Pool(processes=n_jobs)

    for subj_dir in glob.glob('%s/????????' % root):
        mat = ('%s/fMRI/acquisition1/analysis'
               '/model7_HRF_comp_FINAL/SPM.mat' % subj_dir)

        ar = pool.apply_async(load_spm,
                              args=(mat, ),
                              kwds=dict(label=study,
                                        inputs=inputs,
                                        subject=-5,
                                        study=study))

        docs.append(ar)

    pool.close()
    pool.join()

    docs = [doc.get() for doc in docs]

    return fix_docs(docs, contrast_names)


# def get_infos():
#     infos = grr.get_infos()
#     mapping = {}

#     for subject_dir in glob.glob(os.path.join(root, '[A-Z][A-Z]??????')):
#         if os.path.isdir(subject_dir):
#             label = subject_id = os.path.split(subject_dir)[1].lower()
#             mapping[label] = infos.get(
#                 subject_id, {'subject_id': subject_id})

#     return mapping


if __name__ == '__main__':
    # sanitize command-line
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    docs = get_docs(inputs=True)

    execute_glms(docs, output_dir, definitions,
                 dataset_id="vagharchakian2012temporal",
                 )

    # need to resample...
    # import nibabel as nb
    # import numpy as np
    # from nisl import resampling

    # target_affine = np.array([[-3., 0., 0., 78.],
    #                           [0., 3., 0., -111.],
    #                           [0., 0., 3., -51.],
    #                           [0., 0., 0., 1., ]])

    # target_shape = (53, 63, 46)

    # for niimg in glob.glob(os.path.join(
    #         output_dir, study, 'subjects', '*', '*_maps', '*.nii.gz')):
    #     print niimg
    #     img = resampling.resample_img(niimg, target_affine, target_shape)
    #     nb.save(img, niimg)
