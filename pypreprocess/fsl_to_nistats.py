"""
:Module: fsl_to_nistats
:Synopsis: Utility script for converting FSL configuration (design, etc.) files
into Dataframe format.
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import os
import re
import numpy as np
from pypreprocess.external.nistats.design_matrix import make_design_matrix
import pandas as pd

# regex for contrasts
CON_REAL_REGX = ("set fmri\(con_real(?P<con_num>\d+?)\.(?P<ev_num>\d+?)\)"
            " (?P<con_val>\S+)")

# regex for "Number of EVs"
NUM_EV_REGX = """set fmri\(evs_orig\) (?P<evs_orig>\d+)
set fmri\(evs_real\) (?P<evs_real>\d+)
set fmri\(evs_vox\) (?P<evs_vox>\d+)"""

# regex for "Number of contrasts"
NUM_CON_REGX = """set fmri\(ncon_orig\) (?P<ncon>\d+)
set fmri\(ncon_real\) (?P<ncon_real>\d+)"""

# regex for "# EV %i title"
EV_TITLE_REGX = """set fmri\(evtitle\d+?\) \"(?P<evtitle>.+)\""""

# regex for "Title for contrast_real %i"
CON_TITLE_REGX = """set fmri\(conname_real\.\d+?\) \"(?P<conname_real>.+)\""""

# regex for "Basic waveform shape (EV %i)"
# 0 : Square
# 1 : Sinusoid
# 2 : Custom (1 entry per volume)
# 3 : Custom (3 column format)
# 4 : Interaction
# 10 : Empty (all zeros)
EV_SHAPE_REGX = """set fmri\(shape\d+\) (?P<shape>[0|1|3])"""

# regex for "Custom EV file (EV %i)"
EV_CUSTOM_FILE_REGX = """set fmri\(custom\d+?\) \"(?P<custom>.+)\""""


def _get_abspath_relative_to_file(filename, ref_filename):
    """
    Returns the absolute path of a given filename relative to a reference
    filename (ref_filename).

    """

    # we only handle files
    assert os.path.isfile(ref_filename)

    old_cwd = os.getcwd()  # save CWD
    os.chdir(os.path.dirname(ref_filename))  # we're in context now
    abspath = os.path.abspath(filename)  # bing0!
    os.chdir(old_cwd)  # restore CWD

    return abspath


def _insert_directory_in_file_name(filename, directory, level):
    if not isinstance(filename, str):
        return [_insert_directory_in_file_name(x, directory, level)
                for x in filename]

    filename = os.path.abspath(filename)
    parts = filename.split("/")[1:]
    assert level < len(parts)

    head = parts[:-1 - level]
    tail = parts[len(parts) - level - 1:-1]
    return os.path.join("/", *tuple(head + [directory] + tail + [
                os.path.basename(filename)]))


def read_fsl_design_file(design_filename):
    """
    Scrapes an FSL design file for the list of contrasts.

    Returns
    -------
    conditions: list of n_conditions strings
        condition (EV) titles

    timing_files: list of n_condtions strings
        absolute paths of files containing timing info for each condition_id

    contrast_ids: list of n_contrasts strings
        contrast titles

    contrasts: 2D array of shape (n_contrasts, n_conditions)
        array of contrasts, one line per contrast_id; one column per
        condition_id

    Raises
    ------
    AssertionError or IndexError if design_filename is corrupt (not in
    official FSL format)

    """

    # read design file
    design_conf = open(design_filename, 'r').read()

    # scrape n_conditions and n_contrasts
    n_conditions_orig = int(re.search(NUM_EV_REGX,
                                      design_conf).group("evs_orig"))
    n_conditions = int(re.search(NUM_EV_REGX, design_conf).group("evs_real"))
    n_contrasts = int(re.search(NUM_CON_REGX, design_conf).group("ncon_real"))

    # initialize 2D array of contrasts
    contrasts = np.zeros((n_contrasts, n_conditions))

    # lookup EV titles
    conditions = [item.group("evtitle") for item in re.finditer(
                  EV_TITLE_REGX, design_conf)]
    assert len(conditions) == n_conditions_orig

    # lookup contrast titles
    contrast_ids = [item.group("conname_real")for item in re.finditer(
                    CON_TITLE_REGX, design_conf)]
    assert len(contrast_ids) == n_contrasts

    # # lookup EV (condition) shapes
    # condition_shapes = [int(item.group("shape")) for item in re.finditer(
    #         EV_SHAPE_REGX, design_conf)]
    # print(condition_shapes)

    # lookup EV (condition) custom files
    timing_files = [_get_abspath_relative_to_file(item.group("custom"),
                                                  design_filename)
                    for item in re.finditer(EV_CUSTOM_FILE_REGX, design_conf)]

    # lookup the contrast values
    count = 0
    for item in re.finditer(CON_REAL_REGX, design_conf):
        count += 1
        value = float(item.group('con_val'))

        i = int(item.group('con_num')) - 1
        j = int(item.group('ev_num')) - 1

        # roll-call
        assert 0 <= i < n_contrasts, item.group()
        assert 0 <= j < n_conditions, item.group()

        contrasts[i, j] = value

    # roll-call
    assert count == n_contrasts * n_conditions, count

    return conditions, timing_files, contrast_ids, contrasts


def make_paradigm_from_timing_files(timing_files, condition_ids=None):
    if not condition_ids is None:
        assert len(condition_ids) == len(timing_files)

    onsets = []
    durations = []
    amplitudes = []
    _condition_ids = []
    count = 0
    for timing_file in timing_files:
        timing = np.loadtxt(timing_file)
        if timing.ndim == 1:
            timing = timing[np.newaxis, :]

        if condition_ids is None:
            condition_id = os.path.basename(timing_file).lower(
                ).split('.')[0]
        else:
            condition_id = condition_ids[count]
        _condition_ids = _condition_ids + [condition_id
                                           ] * timing.shape[0]

        count += 1

        if timing.shape[1]  == 3:
            onsets = onsets + list(timing[..., 0])
            durations = durations + list(timing[..., 1])
            amplitudes = amplitudes + list(timing[..., 2])
        elif timing.shape[1]  == 2:
            onsets = onsets + list(timing[..., 0])
            durations = durations + list(timing[..., 1])
            amplitudes = durations + list(np.ones(len(timing)))
        elif timing.shape[1] == 1:
            onsets = onsets + list(timing[..., 0])
            durations = durations + list(np.zeros(len(timing)))
            amplitudes = durations + list(np.ones(len(timing)))
        else:
            raise TypeError(
                "Timing info must either be 1D array of onsets of 2D "
                "array with 2 or 3 columns: the first column is for "
                "the onsets, the second for the durations, and the "
                "third --if present-- if for the amplitudes; got %s" % timing)

    return pd.DataFrame({'name': condition_ids,
                         'onset': onsets,
                         'duration': durations,
                         'modulation': amplitudes})


def make_dmtx_from_timing_files(timing_files, condition_ids=None,
                                frametimes=None, n_scans=None, tr=None,
                                add_regs_file=None,
                                add_reg_names=None,
                                **make_dmtx_kwargs):
    # make paradigm
    paradigm = make_paradigm_from_timing_files(timing_files,
                                               condition_ids=condition_ids)

    # make frametimes
    if frametimes is None:
        assert not n_scans is None, ("frametimes not specified, especting a "
                                     "value for n_scans")
        assert not tr is None, ("frametimes not specified, especting a "
                                "value for tr")
        frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
    else:
        assert n_scans is None, ("frametimes specified, not especting a "
                                 "value for n_scans")
        assert tr is None, ("frametimes specified, not especting a "
                                 "value for tr")

    # load addition regressors from file
    if not add_regs_file is None:
        if isinstance(add_regs_file, np.ndarray):
            add_regs = add_regs_file
        else:
            assert os.path.isfile(add_regs_file), (
                "add_regs_file %s doesn't exist")
            add_regs = np.loadtxt(add_regs_file)
        assert add_regs.ndim == 2, (
            "Bad add_regs_file: %s (must contain a 2D array, each column "
            "representing the values of a single regressor)" % add_regs_file)
        if add_reg_names is None:
            add_reg_names = ["R%i" % (col + 1) for col in range(
                    add_regs.shape[-1])]
        else:
            assert len(add_reg_names) == add_regs.shape[1], (
                "Expecting %i regressor names, got %i" % (
                    add_regs.shape[1], len(add_reg_names)))

        make_dmtx_kwargs["add_reg_names"] = add_reg_names
        make_dmtx_kwargs["add_regs"] = add_regs

    # make design matrix
    design_matrix = make_design_matrix(frame_times=frametimes,
                                       paradigm=paradigm,
                                       **make_dmtx_kwargs)

    # return output
    return design_matrix, paradigm, frametimes
