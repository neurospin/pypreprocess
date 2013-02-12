"""
Utilities to download NeuroImaging datasets
"""
# Author: Alexandre Abraham
# License: simplified BSD

import os
import re
import gzip
import glob
import shutil

import numpy as np

from sklearn.datasets.base import Bunch
from external.nisl.datasets import _get_dataset, _fetch_dataset

# definition of consituent files for spm auditory data
SPM_AUDITORY_DATA_FILES = ["fM00223/fM00223_%03i.img" % index
                           for index in xrange(4, 100)]
SPM_AUDITORY_DATA_FILES.append("sM00223/sM00223_002.img")

# definition subject files for haxby dataset
HAXBY_SUBJECT_FILES = ["anat.nii.gz",
                       "bold.nii.gz",
                       "labels.txt",
                       "mask4_vt.nii.gz",
                       "mask8b_face_vt.nii.gz",
                       "mask8b_house_vt.nii.gz",
                       "mask8_face_vt.nii.gz",
                       "mask8_house_vt.nii.gz"]

HAXBY_SUBJECT_IDS = ["subj1",
                     "subj2",
                     "subj3",
                     "subj4"]


def unzip_nii_gz(dirname):
    """
    Helper function for extracting .nii.gz to .nii.

    """

    for filename in glob.glob('%s/*.nii.gz' % dirname):
        if not os.path.exists(re.sub("\.gz", "", filename)):
            f_in = gzip.open(filename, 'rb')
            f_out = open(filename[:-3], 'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            # os.remove(filename)  # XXX why ?


def fetch_poldrack_mixed_gambles(data_dir=None):
    """Download and loads the Poldrack Mixed Gambles dataset
    For the moment, only bold images are loaded
    """
    # definition of dataset files
    file_names = ["ds005/sub0%02i/BOLD/task001_run00%s/bold.nii.gz" % (s, r)
            for s in range(1, 17)
            for r in range(1, 4)]

    # load the dataset
    try:
        # Try to load the dataset
        files = _get_dataset("poldrack_mixed_gambles",
                file_names, data_dir=data_dir)

    except IOError:
        # If the dataset does not exists, we download it
        url = 'http://openfmri.org/system/files/ds005_raw.tgz'
        _fetch_dataset('poldrack_mixed_gambles', [url], data_dir=data_dir)
        files = _get_dataset("poldrack_mixed_gambles",
                file_names, data_dir=data_dir)

    files = np.asarray(np.split(np.asarray(files), 16))

    # return the data
    return Bunch(data=files)


def fetch_openfmri(accession_number, data_dir, redownload=False):
    """ Downloads and extract datasets from www.openfmri.org

        Parameters
        ----------
        accession_number: str
            Dataset identifier, as displayed on https://openfmri.org/data-sets
        data_dir: str
            Destination directory.
        redownload: boolean
            Set to True to force redownload of already available data.
            Defaults to False.

        Datasets
        --------
        {accession_number}: {dataset name}
        ds000001: Balloon Analog Risk-taking Task
        ds000002: Classification learning
        ds000003: Rhyme judgment
        ds000005: Mixed-gambles task
        ds000007: Stop-signal task with spoken & manual responses
        ds000008: Stop-signal task with unselective and selective stopping
        ds000011: Classification learning and tone-counting
        ds000017: Classification learning and stop-signal (1 year test-retest)
        ds000051: Cross-language repetition priming
        ds000052: Classification learning and reversal
        ds000101: Simon task dataset
        ds000102: Flanker task (event-related)
        ds000105: Visual object recognition
        ds000107: Word and object processing

        Returns
        -------
        ds_path: str
            Path of the dataset.
    """

    datasets = {
        'ds000001': 'Balloon Analog Risk-taking Task',
        'ds000002': 'Classification learning',
        'ds000003': 'Rhyme judgment',
        'ds000005': 'Mixed-gambles task',
        'ds000007': 'Stop-signal task with spoken & manual responses',
        'ds000008': 'Stop-signal task with unselective and selective stopping',
        'ds000011': 'Classification learning and tone-counting',
        'ds000017': ('Classification learning and '
                     'stop-signal (1 year test-retest)'),
        'ds000051': 'Cross-language repetition priming',
        'ds000052': 'Classification learning and reversal',
        'ds000101': 'Simon task dataset',
        'ds000102': 'Flanker task (event-related)',
        'ds000105': 'Visual object recognition',
        'ds000107': 'Word and object processing',
    }

    files = {
        'ds000001': ['ds001_raw_fixed_1'],
        'ds000002': ['ds002_raw_0'],
        'ds000003': ['ds003_raw'],
        'ds000005': ['ds005_raw'],
        'ds000007': ['ds007_raw'],
        'ds000008': ['ds008_raw'],
        'ds000011': ['ds011_raw'],
        'ds000017': ['ds017A_raw', 'ds017B_raw'],
        'ds000051': ['ds051_raw'],
        'ds000052': ['ds052_raw'],
        'ds000101': ['ds101_raw'],
        'ds000102': ['ds102_raw'],
        'ds000105': ['ds105_raw'],
        'ds000107': ['ds107_raw'],
    }

    ds_url = 'https://openfmri.org/system/files/%s.tgz'
    ds_name = datasets[accession_number].lower().replace(' ', '_')
    ds_urls = [ds_url % name for name in files[accession_number]]
    ds_path = os.path.join(data_dir, ds_name)

    if not os.path.exists(ds_path) or redownload:
        if os.path.exists(ds_path):
            shutil.rmtree(ds_path)
        _fetch_dataset(ds_name, ds_urls, data_dir, verbose=1)
    return ds_path
