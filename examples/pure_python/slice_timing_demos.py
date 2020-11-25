"""
:Author: DOHMATOB Elvis Dopgima elvis[dot]dohmatob[at]inria[dot]fr
:Synopsis: Manipulations with slice timing
"""

import sys
import os
import warnings
from collections import namedtuple
from tempfile import mkdtemp
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from pypreprocess.slice_timing import STC, fMRISTC
from nilearn.datasets import fetch_nyu_rest
from pypreprocess.datasets import fetch_spm_multimodal_fmri
from pypreprocess.reporting.preproc_reporter import generate_stc_thumbnails

# datastructure for subject data
SubjectData = namedtuple('SubjectData', 'subject_id func output_dir')


def demo_random_brain(output_dir, n_rows=62, n_columns=40, n_slices=10,
                      n_scans=240):
    """Now, how about STC for brain packed with white-noise ? ;)

    """

    print("\r\n\t\t ---demo_random_brain---")

    # populate brain with white-noise (for BOLD values)
    brain_data = np.random.randn(n_rows, n_columns, n_slices, n_scans)

    # instantiate STC object
    stc = STC()

    # fit STC
    stc.fit(raw_data=brain_data)

    # re-slice random brain
    stc.transform(brain_data)

    # QA clinic
    generate_stc_thumbnails([brain_data], [stc.get_last_output_data()],
                            output_dir, close=False)


def demo_sinusoidal_mixture(output_dir, n_slices=10, n_rows=3, n_columns=2,
                          introduce_artefact_in_these_volumes=None,
                          artefact_std=4.,
                          white_noise_std=1e-2):
    """STC for time phase-shifted sinusoidal mixture in the presence of
    white-noise and volume-specific artefacts. This is supposed to be a
    BOLD time-course from a single voxel.

    Parameters
    ----------
    n_slices: int (optional)
        number of slices per 3D volume of the acquisition
    n_rows: int (optional)
        number of rows in simulated acquisition
    n_columns: int (optional)
        number of columns in simmulated acquisition
    white_noise_std: float (optional, default 1e-2)
        amplitude of white noise to add to phase-shifted sample (spatial
        corruption)
    artefact_std: float (optional, default 4)
        amplitude of artefact
    introduce_artefact_in_these_volumesS: string, integer, or list of integers
    (optional, "middle")
        TR/volume index or indices to corrupt with an artefact (a spontaneous
        stray spike, probably due to instability of scanner B-field) of
        amplitude artefact_std

    """

    print("\r\n\t\t ---demo_sinusoid_mixture---")
    slice_indices = np.arange(n_slices, dtype=int)
    timescale = .01
    sine_freq = [.5, .8, .11, .7]

    def my_sinusoid(t):
        """Creates mixture of sinusoids with different frequencies

        """
        res = t * 0
        for f in sine_freq:
            res += np.sin(2 * np.pi * t * f)
        return res

    time = np.arange(0, 24 + timescale, timescale)

    # define timing vars
    freq = 10
    TR = freq * timescale

    # sample the time
    acquisition_time = time[::freq]

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_indices * slice_TR
    shifted_acquisition_time = np.array([tau + acquisition_time
                                     for tau in time_shift])

    # acquire the signal at the corrupt sampled time points
    acquired_signal = np.array([
            [[my_sinusoid(shifted_acquisition_time[j])
              for j in range(n_slices)]
             for _ in range(n_columns)] for _ in range(n_rows)]
                               )

    # add white noise
    acquired_signal += white_noise_std * np.random.randn(
        *acquired_signal.shape)

    n_scans = len(acquisition_time)

    # add artefacts to specific volumes/TRs
    if introduce_artefact_in_these_volumes is None:
        introduce_artefact_in_these_volumes = []
    if isinstance(introduce_artefact_in_these_volumes, int):
        introduce_artefact_in_these_volumes = [
            introduce_artefact_in_these_volumes]
    elif introduce_artefact_in_these_volumes == "middle":
        introduce_artefact_in_these_volumes = [n_scans // 2]
    else:
        assert hasattr(introduce_artefact_in_these_volumes, '__len__')
    introduce_artefact_in_these_volumes = np.array(
        introduce_artefact_in_these_volumes, dtype=int) % n_scans
    acquired_signal[:, :, :, introduce_artefact_in_these_volumes
                ] += artefact_std * np.random.randn(
                    n_rows, n_columns, n_slices,
                    len(introduce_artefact_in_these_volumes))

    # fit STC
    stc = STC()
    stc.fit(n_slices=n_slices, n_scans=n_scans)

    # apply STC
    st_corrected_signal = stc.transform(acquired_signal)

    # QA clinic
    generate_stc_thumbnails(acquired_signal,
                            st_corrected_signal, output_dir, close=False)


def demo_HRF(output_dir, n_slices=10,
             n_rows=2,
             n_columns=3,
             white_noise_std=1e-4,
             ):
    """STC for phase-shifted HRF in the presence of white-noise

    Parameters
    ----------
    n_slices: int (optional, default 21)
        number of slices per 3D volume of the acquisition
    n_voxels_per_slice: int (optional, default 1)
        the number of voxels per slice in the simulated brain
        (setting this to 1 is sufficient for most QA since STC works
        slice-wise, and not voxel-wise)
    white_noise_std: float (optional, default 1e-4)
        STD of white noise to add to phase-shifted sample (spatial corruption)

    """

    print("\r\n\t\t ---demo_HRF---")

    import math

    slice_indices = np.arange(n_slices, dtype=int)

    # create time values scaled at 1%
    timescale = .01
    n_timepoints = 24
    time = np.linspace(0, n_timepoints, num=int(1 + (n_timepoints - 0) / timescale))

    # create gamma functions
    n1 = 4
    lambda1 = 2
    n2 = 7
    lambda2 = 2
    a = .3
    c1 = 1
    c2 = .5

    def compute_hrf(t):
        """Auxiliary function to compute HRF at given times (t)

        """

        hx = (t ** (n1 - 1)) * np.exp(
            -t / lambda1) / ((lambda1 ** n1) * math.factorial(n1 - 1))
        hy = (t ** (n2 - 1)) * np.exp(
            -t / lambda2) / ((lambda2 ** n2) * math.factorial(n2 - 1))

        # create hrf = weighted difference of two gammas
        hrf = a * (c1 * hx - c2 * hy)

        return hrf

    # sample the time and the signal
    freq = 100
    TR = 3
    acquisition_time = time[::TR * freq]
    n_scans = len(acquisition_time)

    # corrupt the sampled time by shifting it to the right
    slice_TR = 1. * TR / n_slices
    time_shift = slice_indices * slice_TR
    shifted_acquisition_time = np.array([tau + acquisition_time
                                     for tau in time_shift])

    # acquire the signal at the corrupt sampled time points
    acquired_sample = np.array([np.vectorize(compute_hrf)(
                shifted_acquisition_time[j])
                                for j in range(n_slices)])
    acquired_sample = np.array([acquired_sample, ] * n_columns)
    acquired_sample = np.array([acquired_sample, ] * n_rows)

    # add white noise
    acquired_sample += white_noise_std * np.random.randn(
        *acquired_sample.shape)

    # fit STC
    stc = STC()
    stc.fit(n_scans=n_scans, n_slices=n_slices)

    # apply STC
    stc.transform(acquired_sample)

    # QA clinic
    generate_stc_thumbnails(acquired_sample,
                            stc.get_last_output_data(), output_dir,
                            close=False)


def _fmri_demo_runner(output_dir, subjects, dataset_id,
                      **spm_slice_timing_kwargs):
    """Demo runner.

    Parameters
    ----------
    subjects: iterable for subject data
        each subject data can be anything, with a func (string or list
        of strings; existing file path(s)) and an output_dir (string,
        existing dir path) field
    dataset_id: string
        a short string describing the data being processed (e.g. "HAXBY!")

    Notes
    -----
    Don't invoke this directly!

    """

    def _load_fmri_data(fmri_files):
        """Helper function to load fmri data from filename / ndarray or list
        of such

        """

        if isinstance(fmri_files, np.ndarray):
            return fmri_files

        if isinstance(fmri_files, str):
            return nibabel.load(fmri_files).get_data()
        else:
            n_scans = len(fmri_files)
            _first = _load_fmri(fmri_files[0])
            data = np.ndarray(tuple(list(_first.shape[:3]
                                         ) + [n_scans]))
            data[..., 0] = _first
            for scan in range(1, n_scans):
                data[..., scan] = _load_fmri(fmri_files[scan])

            return data

    # def _save_output_data(output_data, input_filenames, output_dir):
    #     n_scans = output_data.shape[-1]

    # loop over subjects
    for subject_data in subjects:
        if not os.path.exists(subject_data.output_dir):
            os.makedirs(subject_data.output_dir)

        print("%sSlice-Timing Correction for %s (%s)" % ('\t',
                                                   subject_data.subject_id,
                                                   dataset_id))

        # instantiate corrector
        stc = fMRISTC(**spm_slice_timing_kwargs)

        # fit
        stc.fit(subject_data.func)

        # transform
        stc.transform()

        # plot results
        generate_stc_thumbnails([stc.get_raw_data()],
                                [stc.get_last_output_data()],
                                output_dir, close=False)


def demo_localizer(output_dir):
    output_dir = os.path.join(output_dir, "localizer_output")
    data_path = os.path.join(
        os.environ["HOME"],
        ".nipy", "tests", "data", "s12069_swaloc1_corr.nii.gz")
    if not os.path.exists(data_path):
        warnings.warn("You don't have nipy test data installed!")
        return

    subject_id = 'sub001'
    subject_data = SubjectData(subject_id=subject_id, func=data_path,
                               output_dir=os.path.join(output_dir, subject_id))

    _fmri_demo_runner(output_dir, [subject_data], "localizer")


def demo_spm_multimodal_fmri(output_dir):
    """Demo for SPM multimodal fmri (faces vs scrambled)

    Parameters
    ----------
    output_dir: string
        where output will be written to

    """
    output_dir = os.path.join(output_dir, "spm_multimodal_fmri_output")
    # fetch data
    spm_multimodal_fmri = fetch_spm_multimodal_fmri()

    # subject data factory
    def subject_factory():
        subject_id = "sub001"
        yield SubjectData(subject_id=subject_id,
                          func=spm_multimodal_fmri.func1,
                          output_dir=os.path.join(output_dir, subject_id))

    # invoke demon to run de demo
    _fmri_demo_runner(output_dir, subject_factory(),
                      "SPM Multimodal fMRI faces vs scrambled session 1")


def demo_nyu_rest(output_dir, n_subjects=1):
    """Demo for FSL Feeds data.

    Parameters
    ----------
    data_dir: string, optional
        where the data is located on your disk, where it will be
        downloaded to
    output_dir: string
        where output will be written to

    """
    output_dir = os.path.join(output_dir, "nyu_rest_output")
    # fetch data
    nyu_data = fetch_nyu_rest(n_subjects=n_subjects)

    # subject data factory
    def subject_factory(session=1):
        session_func = [x for x in nyu_data.func if "session%i" % session in x]

        for subject_id in set([os.path.basename(
                os.path.dirname
                (os.path.dirname(x))) for x in session_func]):
            func = [x for x in session_func if subject_id in x]
            assert len(func) == 1
            func = func[0]

            yield SubjectData(subject_id=subject_id, func=func,
                              output_dir=os.path.join(
                                  output_dir, "session%i" % session,
                                  subject_id))

    # invoke demon to run de demo
    _fmri_demo_runner(output_dir, subject_factory(), "NYU Resting State")


if __name__ == '__main__':
    this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    jobfile = os.path.join(this_dir, "multimodal_faces_preproc.ini")

    output_root_dir = None
    if (len(sys.argv) > 1):
        output_root_dir = os.path.abspath(os.path.expanduser(sys.argv[1]))

    if (output_root_dir is None or not os.path.isdir(output_root_dir)):
        output_root_dir = mkdtemp()

    # demo on simulated data
    demo_random_brain(output_root_dir)
    demo_sinusoidal_mixture(output_root_dir)
    demo_HRF(output_root_dir)

    # demo on real data
    demo_localizer(output_root_dir)
    demo_spm_multimodal_fmri(output_root_dir)

    print("output written in {0}".format(output_root_dir))

    plt.show()
