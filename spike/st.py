import os
import sys
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
        for k in xrange(n_slices):
            for f in xrange(l / 2):
                phi[k, f + 1] = -1. * shiftamount[k
                                                  ] * 2 * np.pi / (l / (f + 1))

            # mirror phi about the center
            phi[k, 1 + l / 2 - offset:] = -np.flipud(
                phi[k, 1:l / 2 + offset])

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
        n_scans = raw_data.shape[-1]
        n_slices = raw_data.shape[0]
        l = self._transform.shape[-1]
        if len(raw_data.shape) == 3:
            n_voxels_per_slice = raw_data.shape[1]
        else:
            n_voxels_per_slice = np.prod(raw_data.shape[:2])

        _raw_data = np.zeros((n_slices, n_voxels_per_slice,
                              l))
        _raw_data[:, :, :n_scans] = np.array([self.get_slice_data(raw_data,
                                                                 z)
                                              for z in xrange(n_slices)])

        for z in xrange(n_slices):
            for v in xrange(n_voxels_per_slice):
                _raw_data[z, v, n_scans:] = np.linspace(_raw_data[z, v, n_scans],
                                                        _raw_data[z, v, 0],
                                                        num=l - n_scans,)

        return _raw_data

    def transform(self, raw_data):
        ref_slice = 0
        n_scans = raw_data.shape[-1]
        n_slices = raw_data.shape[2]
        n_rows, n_columns = raw_data.shape[:2]
        l = 2 ** int(np.floor(np.log2(n_scans)) + 1)
        factor = 1. / n_slices
        slices = np.zeros((n_rows, n_columns, n_scans))
        stack = np.zeros((l, n_rows))

        # loop over slices
        self._output_data = 0 * raw_data
        for k in xrange(n_slices):
            # set up time acquired within order
            shiftamount = (k - ref_slice) * factor

            # read slice k data for all volumes
            slices[:, :, :n_scans] = raw_data[:, :, k, :]

            offset = l % 2

            # phi represents a range of phases up to the Nyquist frequency
            phi = np.zeros(l)
            for f in xrange(l / 2):
                phi[f + 1] = -1. * shiftamount * 2 * np.pi / (l / (f + 1))

            # mirror phi about the center
            phi[1 + l / 2 - offset:] = -np.flipud(phi[1:l / 2 + offset])

            # transform phi to frequency domain, then take complex transpose
            shifter = scipy.cos(phi) + scipy.sqrt(-1) * scipy.sin(phi)
            shifter = np.array([shifter
                                for _ in xrange(n_columns)]).T

            # loop over columns of slice k (y axis)
            for i in xrange(n_columns):
                # extract columns from slices
                stack[:n_scans, :] = slices[:, i, :].reshape(
                    (n_rows, n_scans)).T

                # fill-in continuous function to avoid edge effects
                for g in xrange(stack.shape[1]):
                    stack[n_scans:, g] = np.linspace(stack[n_scans - 1, g],
                                                     stack[0, g],
                                                     num=l - n_scans,).T

                # shift the columns of slice k
                stack = np.real(np.fft.ifft(
                        np.fft.fft(stack, axis=0) * shifter, axis=0))

                # re-insert shifted columns of slice k
                slices[:, i, :] = stack[:n_scans, :].T.reshape((n_rows,
                                                                n_scans))

            # re-write slice k for all volumes
            self._output_data[:, :, k, :] = slices[:, :, :n_scans]

        # return output
        return self._output_data

    def get_last_output_data(self):
        return self._output_data

    def get_output_data(self):
        return self.get_last_output_data()


class TestSPMSTC(unittest.TestCase):
    def test_init(self):
        stc = STC()

    def test_fit(self):
        n_slices = 15
        n_scans = 40
        l = 2 ** int(np.floor(np.log2(n_scans)) + 1)
        stc = STC()
        stc.fit(n_slices, n_scans)
        self.assertEqual(stc.get_slice_transform(0).shape, (l,))
        self.assertEqual(stc.get_transform().shape, (n_slices, l))

    def test_sinusoidal_mixture(self):
        n_slices = 4
        n_rows = 1
        n_columns = 1
        n_voxels_per_slice = n_rows * n_columns
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

        # corrupt the sampled time by shifting it to the right
        slice_TR = 1. * TR / n_slices
        time_shift = slice_indices * slice_TR
        shifted_sampled_time = np.array([tau + sampled_time
                                         for tau in time_shift])

        # acquire the signal at the corrupt sampled time points
        acquired_signal = np.array([
                [[my_sinusoid(shifted_sampled_time[j])
                  for j in xrange(n_slices)]
                 for y in xrange(n_columns)] for x in xrange(n_rows)]
                                   )

        # add white noise
        acquired_signal += white_noise_std * np.random.randn(
            *acquired_signal.shape)

        # # add artefacts to specific volumes/TRs
        if introduce_artefact_in_these_volumes is None:
            introduce_artefact_in_these_volumes = []
        # if isinstance(introduce_artefact_in_these_volumes, int):
        #     introduce_artefact_in_these_volumes = [
        #         introduce_artefact_in_these_volumes]
        # elif introduce_artefact_in_these_volumes == "middle":
        #     introduce_artefact_in_these_volumes = [n_scans / 2]
        # else:
        #     assert hasattr(introduce_artefact_in_these_volumes, '__len__')
        # introduce_artefact_in_these_volumes = np.array(
        #     introduce_artefact_in_these_volumes, dtype=int) % n_scans
        # acquired_signal[:, :, introduce_artefact_in_these_volumes
        #                       ] += artefact_std * np.random.randn(
        #     n_slices,
        #     n_voxels_per_slice,
        #     len(introduce_artefact_in_these_volumes))

        # fit STC
        n_scans = len(sampled_time)
        stc = STC()
        stc.fit(n_slices, n_scans)

        # apply STC
        print "Applying full-brain STC transform..."
        st_corrected_signal = stc.transform(acquired_signal)
        print "Done."

        for slice_index in xrange(n_slices):
            for x in xrange(n_rows):
                for y in xrange(n_columns):
                    title = (
                        "Slice-Timing Correction of sampled sine mixeture "
                        "time-course from voxel %s of slice %i \nN.B:- "
                        "TR = %.2f, # slices = %i, # voxels per slice = %i, "
                        "white-noise std = %f, artefact std = %.2f") % (
                        str((x, y)),
                        slice_index, TR, n_slices,
                        n_voxels_per_slice,
                        white_noise_std, artefact_std,
                        )

                    plt.plot(time, signal)
                    plt.hold('on')
                    plt.plot(sampled_time, acquired_signal[x][y][slice_index],
                             'r--o')
                    plt.hold('on')
                    plt.plot(sampled_time,
                             st_corrected_signal[x][y][slice_index],
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
        n_scans = fmri_data.shape[-1]
        n_slices = fmri_data.shape[2]
        stc = STC()
        stc.fit(n_slices, n_scans,
                )

        # do full-brain ST correction
        print "Applying full-brain STC transform..."
        corrected_fmri_data = stc.transform(fmri_data)
        print "Done."

        # save output unto disk
        print "Saving ST corrected image to %s..." % output_filename
        ni.save(ni.Nifti1Image(corrected_fmri_data, fmri_img.get_affine()),
                output_filename)
        print "Done."

        # QA clinic
        sampled_time = np.linspace(0, (n_scans - 1) * TR, n_scans)
        for z in xrange(n_slices):
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
