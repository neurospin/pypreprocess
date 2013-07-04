import numpy as np
import os
import sys
import nibabel
from collections import namedtuple
import matplotlib.pyplot as plt

# pypreproces path
PYPREPROCESS_DIR = os.path.dirname(os.path.split(os.path.abspath(__file__))[0])
sys.path.append(PYPREPROCESS_DIR)

from algorithms.slice_timing.spm_slice_timing import (
    STC, fMRISTC, plot_slicetiming_results)
from external.nisl.datasets import (fetch_nyu_rest, fetch_fsl_feeds_data,
                                    fetch_spm_multimodal_fmri_data)

# datastructure for subject data
SubjectData = namedtuple('SubjectData', 'subject_id func output_dir')


def demo_random_brain(n_rows=62, n_columns=40, n_slices=10, n_scans=240):
    """Now, how about STC for brain packed with white-noise ? ;)

    """

    print "\r\n\t\t ---demo_random_brain---"

    # populate brain with white-noise (for BOLD values)
    brain_data = np.random.randn(n_rows, n_columns, n_slices, n_scans)

    # instantiate STC object
    stc = STC()

    # fit STC
    stc.fit(raw_data=brain_data)

    # re-slice random brain
    stc.transform(brain_data)

    # QA clinic
    plot_slicetiming_results(brain_data, stc.get_last_output_data(),
                             suptitle_prefix="Random brain",)


def demo_sinusoidal_mixture(n_slices=10, n_rows=3, n_columns=2,
                          introduce_artefact_in_these_volumes=None,
                          artefact_std=4.,
                          white_noise_std=1e-2,
                          ):
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

    print "\r\n\t\t ---demo_sinusoid_mixture---"

    slice_indices = np.arange(n_slices, dtype=int)

    timescale = .01
    sine_freq = [.5, .8, .11,
                  .7]  # number of complete cycles per unit time

    def my_sinusoid(t):
        """Creates mixture of sinusoids with different frequencies

        """

        res = t * 0

        for f in sine_freq:
            res += np.sin(2 * np.pi * t * f)

        return res

    time = np.arange(0, 24 + timescale, timescale)
    signal = my_sinusoid(time)

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
              for j in xrange(n_slices)]
             for y in xrange(n_columns)] for x in xrange(n_rows)]
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
        introduce_artefact_in_these_volumes = [n_scans / 2]
    else:
        assert hasattr(introduce_artefact_in_these_volumes, '__len__')
    introduce_artefact_in_these_volumes = np.array(
        introduce_artefact_in_these_volumes, dtype=int) % n_scans
    acquired_signal[:, :, :, introduce_artefact_in_these_volumes
                          ] += artefact_std * np.random.randn(
        n_rows,
        n_columns,
        n_slices,
        len(introduce_artefact_in_these_volumes))

    # fit STC
    stc = STC()
    stc.fit(n_slices=n_slices, n_scans=n_scans)

    # apply STC
    st_corrected_signal = stc.transform(acquired_signal)

    # QA clinic
    plot_slicetiming_results(acquired_signal,
                             st_corrected_signal,
                             TR=TR,
                             ground_truth_signal=signal,
                             ground_truth_time=time,
                             suptitle_prefix="Noisy sinusoidal mixture",
                             )


def demo_HRF(n_slices=10,
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

    print "\r\n\t\t ---demo_HRF---"

    import math

    slice_indices = np.arange(n_slices, dtype=int)

    # create time values scaled at 1%
    timescale = .01
    n_timepoints = 24
    time = np.linspace(0, n_timepoints, num=1 + (n_timepoints - 0) / timescale)

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

    # compute hrf
    signal = compute_hrf(time)

    # sample the time and the signal
    freq = 100
    TR = 3.
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
                                for j in xrange(n_slices)])
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
    plot_slicetiming_results(acquired_sample,
                             stc.get_last_output_data(),
                             TR=TR,
                             ground_truth_signal=signal,
                             ground_truth_time=time,
                             suptitle_prefix="Noisy HRF reconstruction",
                             )


def _fmri_demo_runner(subjects, dataset_id, **spm_slice_timing_kwargs):
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

        if isinstance(fmri_files, basestring):
            return nibabel.load(fmri_files).get_data()
        else:
            n_scans = len(fmri_files)
            _first = _load_fmri_data(fmri_files[0])
            data = np.ndarray(tuple(list(_first.shape[:3]
                                         ) + [n_scans]))
            data[..., 0] = _first
            for scan in xrange(1, n_scans):
                data[..., scan] = _load_fmri_data(fmri_files[scan])

            return data

    def _save_output_data(output_data, input_filenames, output_dir):
        n_scans = output_data.shape[-1]

    # loop over subjects
    for subject_data in subjects:
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
        plot_slicetiming_results(stc.get_raw_data(),
                                 stc.get_last_output_data(),
                                 suptitle_prefix="%s of '%s'" % (
                subject_data.subject_id, dataset_id)
                                 )

        plt.show()


def demo_localizer(output_dir="/tmp/localizer_output"):
    data_path = os.path.join(
        os.environ["HOME"],
        ".nipy/tests/data/s12069_swaloc1_corr.nii.gz")
    if not os.path.exists(data_path):
        raise RuntimeError("You don't have nipy test data installed!")

    subject_id = 'sub001'
    subject_data = SubjectData(subject_id=subject_id, func=data_path,
                               output_dir=os.path.join(output_dir, subject_id))

    _fmri_demo_runner([subject_data], "localizer")


def demo_spm_multimodal_fmri(data_dir="/tmp/spm_multimodal_fmri",
                             output_dir="/tmp/spm_multimodal_fmri_output",
                             ):
    """Demo for SPM multimodal fmri (faces vs scrambled)

    Parameters
    ----------
    data_dir: string, optional
        where the data is located on your disk, where it will be
        downloaded to
    output_dir: string, optional
        where output will be written to

    """

    # fetch data
    spm_multimodal_fmri_data = fetch_spm_multimodal_fmri_data(
        data_dir)

    # subject data factory
    def subject_factory(session=1):
            subject_id = "sub001"

            yield SubjectData(subject_id=subject_id,
                              func=spm_multimodal_fmri_data.func1,
                              output_dir=os.path.join(output_dir, subject_id))

    # invoke demon to run de demo
    _fmri_demo_runner(subject_factory(),
                      "SPM Multimodal fMRI faces vs scrambled session 1")


def demo_nyu_rest(data_dir="/tmp/nyu_data",
                  n_subjects=1,
                  output_dir="/tmp/nyu_rest_output",
                  ):
    """Demo for FSL Feeds data.

    Parameters
    ----------
    data_dir: string, optional
        where the data is located on your disk, where it will be
        downloaded to
    output_dir: string, optional
        where output will be written to

    """

    # fetch data
    nyu_data = fetch_nyu_rest(data_dir=data_dir, n_subjects=1)

    # subject data factory
    def subject_factory(session=1):
        session_func = [x for x in nyu_data.func if "session%i" % session in x]

        for subject_id in set([os.path.basename(
                    os.path.dirname
                    (os.path.dirname(x)))
                               for x in session_func]):
            # set func
            func = [
                x for x in session_func if subject_id in x]
            assert len(func) == 1
            func = func[0]

            yield SubjectData(subject_id=subject_id, func=func,
                              output_dir=os.path.join(
                    output_dir,
                    "session%i" % session, subject_id))

    # invoke demon to run de demo
    _fmri_demo_runner(subject_factory(), "NYU Resting State")


if __name__ == '__main__':
    # # demo on simulated data
    # demo_random_brain()
    # demo_sinusoidal_mixture()
    # demo_HRF()

    # demo on real data
    demo_localizer()
    # demo_spm_multimodal_fmri()  # XXX incorrect assumed slice order!
