import os
import sys
import glob
import nibabel as ni
import scipy
import numpy as np
import unittest
import matplotlib.pyplot as plt


class STC(object):
    def _create_phases(self, n_slices, n_scans,
                       slice_order="ascending",
                       interleaved=False,
                       ref_slice=0,
                       make_power_of_two=True,
                       ):
        slice_indices = np.arange(n_slices)

        # time of acquistion of a single slice
        slice_TR = 1. / n_slices

        # slices will be shifted these amounts respectively
        shiftamount = (slice_indices - ref_slice) * slice_TR

        # l is the least power of 2 not less than n_scans
        l = 2 ** int(np.floor(np.log2(
                    n_scans)) + 1) if make_power_of_two else n_scans
        offset = l % 2

        # phi represents a range of phases up to the Nyquist frequency
        phi = np.zeros((n_slices, l))
        for f in xrange(l / 2):
            phi[:, f + 1] = -1. * shiftamount * 2 * np.pi / (l / (f + 1))

        # mirror phi about the center
        phi[:, 1 + l / 2 - offset:] = np.array(
            [-np.flipud(phi[j, 1:l / 2 + offset])
             for j in xrange(n_slices)])

        # return the phases
        return phi

    def _compute_time_shift_kernel(self, n_slices, n_scans,
                                   slice_order="ascending",
                                   interleaved=False,
                                   ref_slice=0,
                                   make_power_of_two=True,
                                   ):
        # phi represents a range of phases up to the Nyquist frequency
        phi = self._create_phases(n_slices, n_scans,
                                  slice_order=slice_order,
                                  interleaved=interleaved,
                                  ref_slice=ref_slice,
                                  make_power_of_two=make_power_of_two,
                                  )

        # create time shift kernel
        shifter = scipy.cos(phi) + scipy.sqrt(-1.) * scipy.sin(phi)

        return shifter

    def fit(self, n_slices, n_scans):
        self._n_scans = n_scans
        self._n_slices = n_slices
        self._transform = self._compute_time_shift_kernel(n_slices, n_scans)

    def get_slice_data(self, raw_data, z, n_scans=None,):
        n_scans = raw_data.shape[-1]

        if len(raw_data.shape) == 3:
            slice_data = raw_data[z]
        elif len(raw_data.shape) == 4:
            slice_data = raw_data[:, :, z, :]

        # ravel slice_data to 2D array
        slice_data = slice_data.reshape((
                -1,
                 n_scans))

        return slice_data

    def get_slice_transform(self, z):
        return self._transform[z]

    def get_transform(self):
        return self._transform

    def pad_raw_data_with_zeros(self, raw_data):
        if len(raw_data.shape) == 3:
            _raw_data = np.zeros((raw_data.shape[0], raw_data.shape[1],
                                  self._transform.shape[1]))
            _raw_data[:, :, :raw_data.shape[-1]] = raw_data
        elif len(raw_data.shape) == 4:
            _raw_data = np.zeros((raw_data.shape[0], raw_data.shape[1],
                                  raw_data.shape[2],
                                  self._transform.shape[1]))
            _raw_data[:, :, :, :raw_data.shape[-1]] = raw_data

        return _raw_data

    def transform(self, raw_data):
        _raw_data = self.pad_raw_data_with_zeros(raw_data)

        self._output_data = np.array(
            [np.real(np.fft.ifft(np.fft.fft(
                            self.get_slice_data(
                                _raw_data, j)) * self.get_slice_transform(j)))
             for j in xrange(self._n_slices)])

        # trim-off zeros at the tail
        self._output_data = self._output_data[:, :, :self._n_scans]

        # sanitize output shape
        if len(raw_data.shape) == 4:
            # the output has shape (n_slices, n_voxels_per_slice, n_scans)
            # unravel it to match the input data's shape (n_x, n_y, n_slices,
            # n_xcans)
            self._output_data = self._output_data.swapaxes(0, 1).reshape(
                raw_data.shape)

        # return output (just in case caller is eager)
        return self._output_data

    def get_last_output_data(self):
        return self._output_data

    def get_output_data(self):
        return self.get_last_output_data()


class TestSPMSTC(unittest.TestCase):
    def test_init(self):
        stc = STC()

    def test_fit(self):
        n_slices = 12
        n_scans = 40
        l = 2 ** int(np.floor(np.log2(n_scans)) + 1)
        stc = STC()
        stc.fit(n_slices, n_scans)
        self.assertEqual(stc.get_slice_transform(0).shape, (l,))
        self.assertEqual(stc.get_transform().shape, (n_slices, l))

    def test_sinusoidal_mixture(self):
        n_slices = 4
        n_scans = 100
        n_voxels_per_slice = 1
        introduce_artefact_in_these_volumes = None
        artefact_std = 4.
        white_noise_std = 1e-2

        print "\r\n\t\t ---demo_sinusoid---"

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
        sampled_time = time[::freq]
        n_scans = len(sampled_time)

        # corrupt the sampled time by shifting it to the right
        slice_TR = 1. * TR / n_slices
        time_shift = slice_indices * slice_TR
        shifted_sampled_time = np.array([tau + sampled_time
                                         for tau in time_shift])

        # acquire the signal at the corrupt sampled time points
        acquired_signal = np.array([
                [my_sinusoid(shifted_sampled_time[j])
                 for vox in xrange(n_voxels_per_slice)]
                for j in xrange(n_slices)])

        # add white noise
        acquired_signal += white_noise_std * np.random.randn(
            *acquired_signal.shape)

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
        acquired_signal[:, :, introduce_artefact_in_these_volumes
                              ] += artefact_std * np.random.randn(
            n_slices,
            n_voxels_per_slice,
            len(introduce_artefact_in_these_volumes))

        # fit STC
        stc = STC()
        stc.fit(n_slices, n_scans)

        # apply STC
        print "Applying full-brain STC transform..."
        st_corrected_signal = stc.transform(acquired_signal)
        print "Done."

        for slice_index in xrange(n_slices):
            for vox in xrange(n_voxels_per_slice):
                title = ("Slice-Timing Correction of sampled sine mixeture "
                         "time-course from voxel %i of slice %i \nN.B:- "
                         "TR = %.2f, # slices = %i, # voxels per slice = %i, "
                         "white-noise std = %f, artefact std = %.2f, volumes "
                         "corrupt with artefact: %s") % (
                    vox, slice_index, TR, n_slices, n_voxels_per_slice,
                    white_noise_std, artefact_std,
                    ", ".join([str(i)
                               for i in introduce_artefact_in_these_volumes]),
                    )

                plt.plot(time, signal)
                plt.hold('on')
                plt.plot(sampled_time, acquired_signal[slice_index][vox],
                         'r--o')
                plt.hold('on')
                plt.plot(sampled_time, st_corrected_signal[slice_index][vox],
                         's-')
                plt.hold('on')

                # misc
                plt.title(title)
                plt.legend(("Ground-truth signal", "Acquired sample",
                            "ST corrected sample"))
                plt.xlabel("time (s)")
                plt.ylabel("BOLD")

                plt.show()

    def test_spm_auditory(self):
        # path variables
        dataset = "spm-auditory"
        data_dir = "/home/elvis/CODE/datasets/spm_auditory"
        output_dir = "/tmp"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # pypreproces path
        PYPREPROCESS_DIR = os.path.dirname(os.path.split(
                os.path.abspath(__file__))[0])
        sys.path.append(PYPREPROCESS_DIR)
        from datasets_extras import fetch_spm_auditory_data

        _subject_data = fetch_spm_auditory_data(data_dir)

        fmri_img = ni.concat_images(_subject_data['func'],)
        fmri_data = fmri_img.get_data()[:, :, :, 0, :]

        compare_with = ni.concat_images(
            [os.path.join(os.path.dirname(x),
                          "a" + os.path.basename(x))
             for x in _subject_data['func']]).get_data()

        TR = 7.

        output_filename = os.path.join(
            output_dir,
            "st_corrected_" + dataset.rstrip(" ").replace(
                "-", "_") + ".nii.gz",
            )

        print "\r\n\t\t ---demo_BOLD (%s)---" % dataset

        # fit STC
        stc = STC()
        stc.fit(fmri_data.shape[2], fmri_data.shape[-1],
                )

        # do full-brain ST correction
        print "Applying full-brain STC transform..."
        corrected_fmri_data = stc.transform(fmri_data)
        print "Done."

        # # save output unto disk
        # print "Saving ST corrected image to %s..." % output_filename
        # ni.save(ni.Nifti1Image(corrected_fmri_data, fmri_img.get_affine()),
        #         output_filename)
        # print "Done."

        # QA clinic
        n_scans = fmri_data.shape[-1]
        sampled_time = np.linspace(0, (n_scans - 1) * TR, n_scans)
        for z in xrange(7, 13):
            ax1 = plt.subplot2grid((2, 1),
                                   (0, 0))
            # plot acquired sample
            ax1.plot(sampled_time, fmri_data[32][32][z],
                     'r--o')
            ax1.hold('on')

            # plot ST corrected sample
            ax1.plot(sampled_time, corrected_fmri_data[32][32][z],
                     's-')
            ax1.hold('on')
            ax1.plot(sampled_time, compare_with[32][32][z],
                     's-')
            ax1.hold('on')

            # plot ffts
            ax2 = plt.subplot2grid((2, 1),
                                   (1, 0))

            ax2.plot(sampled_time[1:],
                     np.abs(np.fft.fft(fmri_data[32][32][z])[1:]))

            ax2.plot(sampled_time[1:],
                     np.abs(np.fft.fft(corrected_fmri_data[32][32][z])[1:]))

            ax2.plot(sampled_time[1:],
                     np.abs(np.fft.fft(compare_with[32][32][z])[1:]))

            # misc
            ax1.legend(("Acquired sample",
                        "STC method 1",
                        "STC method 2"))
            ax2.legend(("Acquired sample",
                        "STC method 1",
                        "STC method 2"))
            ax1.set_xlabel("time (s)")
            ax1.set_ylabel("BOLD")

            plt.show()


if __name__ == '__main__':
    unittest.main()
