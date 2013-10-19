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
SPM_LOADER_DIR = os.path.join(
    os.path.dirname(os.path.split(
            os.path.abspath(__file__))[0]),
    "spm_loader")
sys.path.append(SPM_LOADER_DIR)

from spm_loader.spm import load_intra as load_spm
from spm_loader.utils import fix_docs, execute_glms

study = 'cauvet2009muslang'
root = ('/neurospin/unicog/protocols/IRMf'
        '/Muslang_hara_cauvet_pallier_2009.new/Subjects')
output_dir = '/tmp/protocols'


contrast_names = {
    'level1 (more struct) lang': 'c16_lang',
    'level2 lang': 'c08_lang',
    'level3 lang': 'c04_lang',
    'level4 lang': 'c02_lang',
    'level5 (less struct) lang': 'c01_lang',
    'level1 (more struct) music': 'c16_music',
    'level2 music': 'c08_music',
    'level3 music': 'c04_music',
    'level4 music': 'c02_music',
    'level5 (less struct) music': 'c01_music',
    'sentence - word': 'sentence - word',
    'word - sentence': 'word - sentence',
    }

definitions = {
    'auditory sentences more vs less structure': {
        'c16_lang': 1,
        'c08_lang': 1,
        'c04_lang': 0,
        'c02_lang': -1,
        'c01_lang': -1,
        },
    'auditory sentences less vs more structure': {
        'c16_lang': -1,
        'c08_lang': -1,
        'c04_lang': 0,
        'c02_lang': 1,
        'c01_lang': 1,
        },
    'auditory music more vs less structure': {
        'c16_music': 1,
        'c08_music': 1,
        'c04_music': 0,
        'c02_music': -1,
        'c01_music': -1,
        },
    'auditory music less vs more structure': {
        'c16_music': -1,
        'c08_music': -1,
        'c04_music': 0,
        'c02_music': 1,
        'c01_music': 1,
        },

    'auditory structured sentences vs rest': {
        'c16_lang': 1,
        'c08_lang': 1,
        'c04_lang': 0,
        'c02_lang': 0,
        'c01_lang': 0,
        },
    'auditory unstructured sentences vs rest': {
        'c16_lang': 0,
        'c08_lang': 0,
        'c04_lang': 0,
        'c02_lang': 1,
        'c01_lang': 1,
        },
    'auditory structured music vs rest': {
        'c16_music': 1,
        'c08_music': 1,
        'c04_music': 0,
        'c02_music': 0,
        'c01_music': 0,
        },
    'auditory unstructured music vs rest': {
        'c16_music': 0,
        'c08_music': 0,
        'c04_music': 0,
        'c02_music': 1,
        'c01_music': 1,
        },
    }


def get_docs(inputs=False):
    n_jobs = 24

    docs = []
    pool = multiprocessing.Pool(processes=n_jobs)

    for subj_dir in glob.glob('%s/suj??' % root):
        mat = ('%s/fMRI/acquisition1/analysis'
               '/hrf_deriv/SPM.mat' % subj_dir)

        ar = pool.apply_async(load_spm,
                              args=(mat, ),
                              kwds=dict(label=study,
                                        subject=-5,
                                        inputs=inputs,
                                        study=study))

        docs.append(ar)

    pool.close()
    pool.join()

    docs = [doc.get() for doc in docs]

    return fix_docs(docs, contrast_names)


if __name__ == '__main__':
    # sanitize command-line
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    docs = get_docs(inputs=True)

    execute_glms(docs, output_dir, definitions,
                 dataset_id="cauvet2009muslang",
                 # do_preproc=False,
                 # smoothed=5.,
                 )
