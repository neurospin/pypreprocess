import os
import glob
import re
import nibabel
from sklearn.utils import Bunch

# XXX nilearn.datasets.py got factorized recently. The following codeblock
# is to ensure backward compat.
try:
    from nilearn.datasets.utils import (_fetch_file, _fetch_files,
                                        _uncompress_file, _get_dataset_dir)
except ImportError:
    # old version, or change not yet in release
    from nilearn.datasets import (_fetch_file, _fetch_files,
                                  _uncompress_file, _get_dataset_dir)

SPM_AUDITORY_DATA_FILES = ["fM00223/fM00223_%03i.img" % index
                           for index in range(4, 100)]
FSL_FEEDS_DATA_FILES = ["fmri.nii.gz", "structural_brain.nii.gz"]
SPM_AUDITORY_DATA_FILES.append("sM00223/sM00223_002.img")


def fetch_spm_auditory(data_dir=None, data_name='spm_auditory',
                       subject_id="sub001", verbose=1):
    """Function to fetch SPM auditory single-subject data.

    Parameters
    ----------
    data_dir: string
        path of the data directory. Used to force data storage in a specified
        location. If the data is already present there, then will simply
        glob it.

    Returns
    -------
    data: sklearn.utils.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func': string list. Paths to functional images
        - 'anat': string list. Path to anat image

    References
    ----------
    :download:
        http://www.fil.ion.ucl.ac.uk/spm/data/auditory/

    """
    data_dir = _get_dataset_dir(data_name, data_dir=data_dir,
                                verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)

    def _glob_spm_auditory_data():
        """glob data from subject_dir.

        """

        if not os.path.exists(subject_dir):
            return None

        subject_data = {}
        for file_name in SPM_AUDITORY_DATA_FILES:
            file_path = os.path.join(subject_dir, file_name)
            if os.path.exists(file_path):
                subject_data[file_name] = file_path
            else:
                print("%s missing from filelist!" % file_name)
                return None

        _subject_data = dict(data_dir=data_dir)
        _subject_data["func"] = sorted([subject_data[x]
                                        for x in subject_data.keys()
                                        if re.match("^fM00223_0\d\d\.img$",
                                                    os.path.basename(x))])

        # volumes for this dataset of shape (64, 64, 64, 1); let's fix this
        for x in _subject_data["func"]:
            vol = nibabel.load(x)
            if len(vol.shape) == 4:
                vol = nibabel.Nifti1Image(vol.get_data()[:, :, :, 0],
                                          vol.get_affine())
                nibabel.save(vol, x)

        _subject_data["anat"] = [subject_data[x] for x in subject_data.keys()
                                 if re.match("^sM00223_002\.img$",
                                             os.path.basename(x))][0]

        # ... same thing for anat
        vol = nibabel.load(_subject_data["anat"])
        if len(vol.shape) == 4:
            vol = nibabel.Nifti1Image(vol.get_data()[:, :, :, 0],
                                      vol.get_affine())
            nibabel.save(vol, _subject_data["anat"])

        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_spm_auditory_data()
    if not data is None:
        return data

    # No. Download the data
    print("Data absent, downloading...")
    url = ("http://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/"
           "MoAEpilot.zip")
    archive_path = os.path.join(subject_dir, os.path.basename(url))
    for i in range(2):
        _fetch_file(url, subject_dir)
        try:
            _uncompress_file(archive_path)
            break
        except IOError:
            if i == 0:
                print("Archive corrupted, trying to download it again.")
                os.remove(archive_path)
            else:
                raise IOError("Unable to download archive from %s" % url)
    return _glob_spm_auditory_data()


def fetch_fsl_feeds(data_dir=None, data_name="fsl_feeds", verbose=1):
    """Function to fetch FSL FEEDS dataset (single-subject)

    Parameters
    ----------
    data_dir: string
        path of the data directory. Used to force data storage in a specified
        location. If the data is already present there, then will simply
        glob it.

    Returns
    -------
    data: sklearn.utils.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func': string list. Paths to functional images
        - 'anat': string list. Path to anat image

    """
    data_dir = _get_dataset_dir(data_name, data_dir=data_dir,
                                verbose=verbose)

    def _glob_fsl_feeds_data(subject_dir):
        """glob data from subject_dir.

        """

        if not os.path.exists(subject_dir):
            return None

        for file_name in FSL_FEEDS_DATA_FILES:
            file_path = os.path.join(subject_dir, file_name)
            if os.path.exists(file_path) or os.path.exists(
                    file_path.rstrip(".gz")):
                file_name = re.sub("(?:\.nii\.gz|\.txt)", "", file_name)
            else:
                if not os.path.basename(subject_dir) == 'data':
                    return _glob_fsl_feeds_data(os.path.join(subject_dir,
                                                             'feeds/data'))
                else:
                    print("%s missing from filelist!" % file_name)
                    return None
        return Bunch(data_dir=data_dir,
                     func=os.path.join(subject_dir, "fmri.nii.gz"),
                     anat=os.path.join(
                         subject_dir, "structural_brain.nii.gz"))

    # maybe data_dir already contents the data ?
    data = _glob_fsl_feeds_data(data_dir)
    if not data is None:
        return data

    # download the data
    print("Data absent, downloading...")
    url = ("http://fsl.fmrib.ox.ac.uk/fsldownloads/oldversions/"
           "fsl-4.1.0-feeds.tar.gz")
    archive_path = os.path.join(data_dir, os.path.basename(url))
    for i in range(2):
        _fetch_files(
            data_dir, [("feeds", url, {"uncompress": True, "move": "fsl.tar"})]
        )        
    return _glob_fsl_feeds_data(data_dir)


def fetch_spm_multimodal_fmri(data_dir=None, data_name="spm_multimodal_fmri",
                              subject_id="sub001", verbose=1):
    """Fetcher for Multi-modal Face Dataset.

    Parameters
    ----------
    data_dir: string
        path of the data directory. Used to force data storage in a specified
        location. If the data is already present there, then will simply
        glob it.

    Returns
    -------
    data: sklearn.utils.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func1': string list. Paths to functional images for session 1
        - 'func2': string list. Paths to functional images for session 2
        - 'trials_ses1': string list. Path to onsets file for session 1
        - 'trials_ses2': string list. Path to onsets file for session 2
        - 'anat': string. Path to anat file

    References
    ----------
    :download:
        http://www.fil.ion.ucl.ac.uk/spm/data/mmfaces/

    """

    data_dir = _get_dataset_dir(data_name, data_dir=data_dir,
                                verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)

    def _glob_spm_multimodal_fmri_data():
        """glob data from subject_dir."""
        _subject_data = dict(data_dir=data_dir, slice_order='descending')

        for s in range(2):
            # glob func data for session s + 1
            session_func = sorted(glob.glob(
                os.path.join(
                    subject_dir,
                    ("fMRI/Session%i/fMETHODS-000%i-*-01.img" % (
                        s + 1, s + 5)))))
            if len(session_func) < 390:
                print("Missing %i functional scans for session %i." % (
                    390 - len(session_func), s))
                return None
            else:
                _subject_data['func%i' % (s + 1)] = session_func

            # glob trials .mat file
            sess_trials = os.path.join(
                subject_dir,
                "fMRI/trials_ses%i.mat" % (s + 1))
            if not os.path.isfile(sess_trials):
                print("Missing session file: %s" % sess_trials)
                return None
            else:
                _subject_data['trials_ses%i' % (s + 1)] = sess_trials

        # glob for anat data
        anat = os.path.join(subject_dir, "sMRI/smri.img")
        if not os.path.isfile(anat):
            print("Missing structural image.")
            return None
        else:
            _subject_data["anat"] = anat

        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_spm_multimodal_fmri_data()
    if not data is None:
        return data

    # No. Download the data
    print("Data absent, downloading...")
    urls = [
        # fmri
        ("http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/"
        "multimodal_fmri.zip"),

        # structural
        ("http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/"
         "multimodal_smri.zip")
        ]

    for url in urls:
        archive_path = os.path.join(subject_dir, os.path.basename(url))
        for i in range(2):
            _fetch_file(url, subject_dir)
            try:
                _uncompress_file(archive_path)
                break
            except:
                if i == 0:
                    print("Archive corrupted, trying to download it again.")
                    os.remove(archive_path)
                else:
                    raise IOError("Unable to download archive from %s" % url)

    return _glob_spm_multimodal_fmri_data()


def fetch_openfmri(data_dir, dataset_id, force_download=False, verbose=1):
    files = {
        'ds001': ['ds001_raw'],
        'ds002': ['ds002_raw'],
        'ds003': ['ds003_raw'],
        'ds005': ['ds005_raw'],
        'ds006A': ['ds006A_raw'],
        'ds007': ['ds007_raw'],
        'ds008': ['ds008_raw'],
        'ds011': ['ds011_raw'],
        'ds017A': ['ds017A_raw'],
        'ds017B': ['ds017B_raw'],
        'ds051': ['ds051_raw'],
        'ds052': ['ds052_raw'],
        'ds101': ['ds101_raw'],
        'ds102': ['ds102_raw'],
        'ds105': ['ds105_raw'],
        'ds107': ['ds107_raw'],
        'ds108': ['ds108_raw_part1', 'ds108_raw_part2', 'ds108_raw_part3'],
        'ds109': ['ds109_raw'],
        'ds110': ['ds110_raw_part1', 'ds110_raw_part2', 'ds110_raw_part3',
                  'ds110_raw_part4', 'ds110_raw_part5', 'ds110_raw_part6']
        }

    if dataset_id not in files:
        raise Exception('Unknown dataset %s' % dataset_id)

    base_url = 'http://openfmri.s3.amazonaws.com/tarballs/%s.tgz'
    urls = [(dataset_id, base_url % f, {'uncompress':True}) for f in files[dataset_id]]
    temp_dir = os.path.join(data_dir, '_%s' % dataset_id, dataset_id)
    output_dir = os.path.join(data_dir, dataset_id)
    if not os.path.exists(output_dir) and not force_download:
        _fetch_files(data_dir, urls, verbose=verbose)
    return output_dir
