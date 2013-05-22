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
import joblib
import numpy as np

from nipype.interfaces.base import Bunch

from external.nisl.datasets import _get_dataset, _fetch_dataset, \
    _uncompress_file, _fetch_file

# definition of consituent files for spm auditory data
SPM_AUDITORY_DATA_FILES = ["fM00223/fM00223_%03i.img" % index
                           for index in xrange(4, 100)]
SPM_AUDITORY_DATA_FILES.append("sM00223/sM00223_002.img")

FSL_FEEDS_DATA_FILES = ["fmri.nii.gz", "structural_brain.nii.gz"]

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


def _glob_spm_auditory_data(subject_dir):
    if not os.path.exists(subject_dir):
        return None

    subject_data = dict()
    subject_data["subject_dir"] = subject_dir
    for file_name in SPM_AUDITORY_DATA_FILES:
        file_path = os.path.join(subject_dir, file_name)
        if os.path.exists(file_path) or os.path.exists(
            file_path.rstrip(".gz")):
            file_name = re.sub("(?:\.nii\.gz|\.txt)", "", file_name)
            subject_data[file_name] = file_path
        else:
            print "%s missing from filelist!" % file_name
            return None

    _subject_data = {}
    _subject_data["func"] = [subject_data[x] for x in subject_data.keys()
                             if re.match("^fM00223_0\d\d\.img$",
                                         os.path.basename(x))]
    _subject_data["func"].sort()

    _subject_data["anat"] = [subject_data[x] for x in subject_data.keys()
                             if re.match("^sM00223_002\.img$",
                                         os.path.basename(x))][0]
    return _subject_data


def _glob_fsl_feeds_data(subject_dir):
    if not os.path.exists(subject_dir):
        return None

    subject_data = dict()
    subject_data["subject_dir"] = subject_dir
    for file_name in FSL_FEEDS_DATA_FILES:
        file_path = os.path.join(subject_dir, file_name)
        if os.path.exists(file_path) or os.path.exists(
            file_path.rstrip(".gz")):
            file_name = re.sub("(?:\.nii\.gz|\.txt)", "", file_name)
            subject_data[file_name] = file_path
        else:
            if not os.path.basename(subject_dir) == 'data':
                return _glob_fsl_feeds_data(os.path.join(subject_dir,
                                                         'feeds/data'))
            else:
                print "%s missing from filelist!" % file_name
                return None

    _subject_data = {"func": os.path.join(subject_dir,
                                          "fmri.nii"),
                     "anat": os.path.join(subject_dir,
                                         "structural_brain.nii"),
                     }

    return _subject_data


def fetch_haxby_subject_data(data_dir, subject_id, url, redownload=False):
    archive_name = os.path.basename(url)
    archive_path = os.path.join(data_dir, archive_name)
    subject_dir = os.path.join(data_dir, subject_id)
    if redownload:
        try:
            print "Zapping all old downloads .."
            shutil.rmtree(subject_dir)
            os.remove(archive_path)
        except OSError:
            pass
        finally:
            print "Done."
    if os.path.exists(subject_dir):
        subject_data = _glob_haxby_subject_data(subject_dir)
        if subject_data is None:
            shutil.rmtree(subject_dir)
            return fetch_haxby_subject_data(data_dir, subject_id, url)
        else:
            return subject_id, subject_data
    elif os.path.exists(archive_path):
        try:
            _uncompress_file(archive_path)
        except:
            print "Archive corrupted, trying to download it again."
            os.remove(archive_path)
            return fetch_haxby_subject_data(data_dir, subject_id, url)
    else:
        _fetch_file(url, data_dir)
        try:
            _uncompress_file(archive_path)
        except:
            print "Archive corrupted, trying to download it again."
            os.remove(archive_path)
            return fetch_haxby_subject_data(data_dir, subject_id, url)
        return subject_id, _glob_haxby_subject_data(subject_dir)


def _glob_haxby_subject_data(subject_dir):
    if not os.path.exists(subject_dir):
        return None

    subject_data = dict()
    subject_data["subject_dir"] = subject_dir
    for file_name in HAXBY_SUBJECT_FILES:
        file_path = os.path.join(subject_dir, file_name)
        if os.path.exists(file_path) or os.path.exists(
            file_path.rstrip(".gz")):
            file_name = re.sub("(?:\.nii\.gz|\.txt)", "", file_name)
            subject_data[file_name] = file_path
        else:
            print "%s missing from filelist!" % file_name
            return None

    return Bunch(subject_data)


def fetch_haxby(data_dir=None, subject_ids=None, redownload=False,
                n_jobs=1):
    """
    Download and loads the haxby dataset

    Parameters
    ----------
    data_dir: string, optional
    Path of the data directory. Used to force data storage in a specified
    location. Default: None

    subject_ids: list of string, option
    Only download data for these subjects

    redownload: bool, optional
    Delete all local file copies on disk and re-download

    Returns
    -------
    data: dictionary, keys are subject ids (subj1, subj2, etc.)
    'bold': string
    Path to nifti file with bold data
    'session_target': string
    Path to text file containing session and target data
    'mask*': string
    Path to correspoding nifti mask file
    'labels': string
    Path to text file containing labels (can be used for LeaveOneLabelOut
    cross validation for example)

    References
    ----------
    `Haxby, J., Gobbini, M., Furey, M., Ishai, A., Schouten, J.,
    and Pietrini, P. (2001). Distributed and overlapping representations of
    faces and objects in ventral temporal cortex. Science 293, 2425-2430.`

    Notes
    -----
    PyMVPA provides a tutorial using this dataset :
    http://www.pymvpa.org/tutorial.html

    More informations about its structure :
    http://dev.pymvpa.org/datadb/haxby2001.html

    See `additional information
    <http://www.sciencemag.org/content/293/5539/2425>`_
    """

    data_dir = os.path.join(data_dir, "haxby2001")

    subjects = dict()
    if subject_ids is None:
        subject_ids = HAXBY_SUBJECT_IDS
    else:
        subject_ids = [subject_id for subject_id in subject_ids \
                           if subject_id in HAXBY_SUBJECT_IDS]

    # url spitter
    def url_factory():
        for subject_id in subject_ids:
            url = ('http://data.pymvpa.org'
                   '/datasets/haxby2001/%s-2010.01.14.tar.gz') % subject_id
            yield data_dir, subject_id, url, redownload

    # parallel fetch
    pairs = joblib.Parallel(n_jobs=n_jobs, verbose=100)(
        joblib.delayed(fetch_haxby_subject_data)(x, y, z, w)\
            for x, y, z, w in url_factory())

    # pack pairs in to a dict
    for subject_id, subject_data in pairs:
        subjects[subject_id] = subject_data

    return subjects


def fetch_spm_auditory_data(data_dir, redownload=False):
    '''
Function to fetch SPM auditory data.

'''

    url = "ftp://ftp.fil.ion.ucl.ac.uk/spm/data/MoAEpilot/MoAEpilot.zip"
    subject_dir = data_dir
    archive_path = os.path.join(subject_dir, os.path.basename(url))
    if redownload:
        try:
            print "Zapping all old downloads .."
            # shutil.rmtree(subject_dir)
            # os.remove(archive_path)
        except OSError:
            pass
        finally:
            print "Done."
    if os.path.exists(subject_dir):
        subject_data = _glob_spm_auditory_data(subject_dir)
        if subject_data is None:
            # shutil.rmtree(subject_dir)
            return fetch_spm_auditory_data(data_dir)
        else:
            return subject_data
    elif os.path.exists(archive_path):
        try:
            _uncompress_file(archive_path)
        except:
            print "Archive corrupted, trying to download it again."
            # os.remove(archive_path)
            return fetch_spm_auditory_data(data_dir)
    else:
        _fetch_file(url, data_dir)
        try:
            _uncompress_file(archive_path)
        except:
            print "Archive corrupted, trying to download it again."
            # os.remove(archive_path)
            return fetch_spm_auditory_data(data_dir)
        return _glob_spm_auditory_data(subject_dir)


def fetch_fsl_feeds_data(data_dir, redownload=False):
    '''
    Function to fetch SPM auditory data.

    '''

    url = ("http://fsl.fmrib.ox.ac.uk/fsldownloads/oldversions/"
           "fsl-4.1.0-feeds.tar.gz")
    subject_dir = data_dir
    archive_path = os.path.join(subject_dir, os.path.basename(url))
    if redownload:
        try:
            print "Zapping all old downloads .."
            shutil.rmtree(subject_dir)
            os.remove(archive_path)
        except OSError:
            pass
        finally:
            print "Done."
    if os.path.exists(subject_dir):
        subject_data = _glob_fsl_feeds_data(subject_dir)
        if subject_data is None:
            shutil.rmtree(subject_dir)
            return fetch_fsl_feeds_data(data_dir)
        else:
            return subject_data
    elif os.path.exists(archive_path):
        try:
            _uncompress_file(archive_path)
        except:
            print "Archive corrupted, trying to download it again."
            os.remove(archive_path)
            return fetch_fsl_feeds_data(data_dir)
    else:
        _fetch_file(url, data_dir)
        try:
            _uncompress_file(archive_path)
        except:
            print "Archive corrupted, trying to download it again."
            os.remove(archive_path)
            return fetch_fsl_feeds_data(data_dir)
        return _glob_fsl_feeds_data(subject_dir)


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
