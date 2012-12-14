""" Additional datasets for Nisl

As some dataset are local to Neurospin or requires authentication,
they cannot be shipped with Nisl public version. This file is meant
to provide those dataset inteded for internal use.

From parietal_python commit 40e4692dc0798eed3d7effb3a151f763c591492b
"""

import os
import collections

import numpy as np
from . import Bunch

from .datasets import _get_dataset, _get_dataset_dir


def _filter_column(array, col, criteria):
    """ Return index array matching criteria

    Parameters
    ----------

    array: numpy array with columns
        Array in which data will be filtered

    col: string
        Name of the column

    criteria: integer (or float), pair of integers, string or list of these
        if integer, select elements in column matching integer
        if a tuple, select elements between the limits given by the tuple
        if a string, select elements that match the string
    """
    # Raise an error if the column does not exist
    array[col]

    if not isinstance(criteria, basestring) and \
            not isinstance(criteria, tuple) and \
            isinstance(criteria, collections.Iterable):
        filter = np.zeros(array.shape, dtype=np.bool)
        for criterion in criteria:
            filter = np.logical_or(filter,
                        _filter_column(array, col, criterion))
        return filter

    if isinstance(criteria, tuple):
        if len(criteria) != 2:
            raise ValueError("An interval must have 2 values")
        if criteria[0] is None:
            return array[col] <= criteria[1]
        if criteria[1] is None:
            return array[col] >= criteria[0]
        filter = array[col] <= criteria[1]
        return np.logical_and(filter, array[col] >= criteria[0])

    return array[col] == criteria


def _filter_columns(array, filters):
    filter = np.ones(array.shape, dtype=np.bool)
    for column in filters:
        filter = np.logical_and(filter,
                _filter_column(array, column, filters[column]))
    return filter


def _remove_end_digit(site):
    if site.endswith('1') or site.endswith('2'):
        return site[:-2]


def fetch_abide(data_dir=None, verbose=0,
                **kwargs):
    """ Load ABIDE dataset

    The ABIDE dataset must be installed in the data_dir (or NISL_DATA env var)
    into an 'ABIDE' folder. The Phenotypic information file should be in this
    folder too.

    Parameters
    ----------

    SUB_ID: list of integers in [50001, 50607], optional
        Ids of the subjects to be loaded.

    DX_GROUP: integer in {1, 2}, optional
        1 is autism, 2 is control

    DSM_IV_TR: integer in [0, 4], optional
        O is control, 1 is autism, 2 is Asperger, 3 is PPD-NOS,
        4 is Asperger or PPD-NOS

    AGE_AT_SCAN: float in [6.47, 64], optional
        Age of the subject

    SEX: integer in {1, 2}, optional
        1 is male, 2 is female

    HANDEDNESS_CATEGORY: string in {'R', 'L', 'Mixed', 'Ambi'}, optional
        R = Right, L = Left, Ambi = Ambidextrous

    HANDEDNESS_SCORE: integer in [-100, 100], optional
        Positive = Right, Negative = Left, 0 = Ambidextrous
    """

    name_csv = 'Phenotypic_V1_0b.csv'
    dataset_dir = _get_dataset_dir('ABIDE', data_dir=data_dir)
    path_csv = _get_dataset('ABIDE', [name_csv], data_dir=data_dir)[0]

    pheno = np.genfromtxt(path_csv, names=True, delimiter=',', dtype=None)
    filter = _filter_columns(pheno, kwargs)
    pheno = pheno[filter]

    # Get the files for all remaining subjects
    folders = [_remove_end_digit(site) + '_' + str(id) for (site, id)
            in pheno[['SITE_ID', 'SUB_ID']]]

    anat = []
    func = []
    for folder in folders:
        base = os.path.join(dataset_dir, folder, folder, 'scans')
        fanat = os.path.join(base, 'anat', 'resources', 'NIfTI', 'files',
                            'mprage.nii')
        ffunc = os.path.join(base, 'rest', 'resources', 'NIfTI', 'files',
                            'rest.nii')
        if (os.path.exists(fanat), os.path.exists(ffunc)):
            anat.append(fanat)
            func.append(ffunc)

    # return the data
    return Bunch(pheno=pheno, func=func, anat=anat)
