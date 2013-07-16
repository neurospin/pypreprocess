import numpy as np
import nibabel
import os
import inspect
from ..spm_realign import (_load_vol,
                           _load_specific_vol,
                           compute_rate_of_change_of_chisq,
                           MRIMotionCorrection)

# global setup
OUTPUT_DIR = "/tmp/test_spm_realign_data_dir"
IMAGE_EXTENSIONS = [".nii", ".nii.gz", ".img"]


def create_random_image(shape=None,
                        ndim=3,
                        n_scans=None,
                        affine=np.eye(4),
                        parent_class=nibabel.Nifti1Image):

    if shape is None:
        shape = np.random.random_integers(20, size=ndim)

    ndim = len(shape)
    if not n_scans is None and ndim == 4:
        shape[-1] = n_scans

    return parent_class(np.random.randn(*shape), affine)


def test_load_vol():
    """
    Tests the loading of a 3D vol by filename (.nii, .nii.gz, .img, etc.)
    or from a nibabel image object (Nifti1Image, Nifti1Pair, etc.)

    """

    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # creat a volume
    vol = create_random_image()

    # test loading vol from nibabel object
    _vol = _load_vol(vol)
    assert isinstance(_vol, type(vol))
    assert _vol.shape == vol.shape
    assert np.all(_vol.get_data() == vol.get_data())

    # test loading vol by filename
    for ext in IMAGE_EXTENSIONS:
        # save vol with extension ext
        vol_filename = os.path.join(output_dir, "vol%s" % ext)
        nibabel.save(vol, vol_filename)

        # note that .img loads as Nifti1Pair, not Nifti1Image
        vol_type = nibabel.Nifti1Pair if ext == '.img' else nibabel.Nifti1Image

        # load the vol by filename
        _vol = _load_vol(vol_filename)
        assert isinstance(_vol, vol_type)
        assert _vol.shape == vol.shape
        assert np.all(_vol.get_data() == vol.get_data())


def test_load_specific_vol():
    """
    Tests the loading of a single specific 3D vol from a 4D film, by
    filename(s) (.nii, .nii.gz, .img, etc.) or from a nibabel image object
    (Nifti1Image, Nifti1Pair, etc.).

    """

    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    n_scans = 23

    # create 4D film
    film = create_random_image(ndim=4, n_scans=n_scans)

    # test loading vol from nibabel image object
    for t in xrange(n_scans):
        _vol, _n_scans = _load_specific_vol(film, t)
        assert _n_scans == n_scans
        assert isinstance(_vol, type(film))
        assert _vol.shape == film.shape[:-1]
        assert np.all(_vol.get_data() == film.get_data()[..., t])

    # test loading vol from a single 4D filename
    for ext in IMAGE_EXTENSIONS:
        for film_filename_type in ['str', 'list']:
            if film_filename_type == 'str':
                # save film as single filename with extension ext
                film_filename = os.path.join(output_dir, "4D%s" % ext)
                nibabel.save(film, film_filename)
            else:
                # save film as multiple filenames (3D vols), with ext extension
                vols = nibabel.four_to_three(film)
                film_filename = []
                for t, vol in zip(xrange(n_scans), vols):
                    vol_filename = os.path.join(output_dir,
                                                "vol_%i%s" % (t, ext))
                    nibabel.save(vol, vol_filename)
                    film_filename.append(vol_filename)

            # test loading proper
            for t in xrange(n_scans):
                # note that .img loads as Nifti1Pair, not Nifti1Image
                vol_type = nibabel.Nifti1Pair if ext == '.img' else \
                    nibabel.Nifti1Image

                # load specific 3D vol from 4D film by filename
                _vol, _n_scans = _load_specific_vol(film_filename, t)
                assert _n_scans == n_scans
                assert isinstance(_vol, vol_type)
                assert _vol.shape == film.shape[:-1]
                assert np.all(_vol.get_data() == film.get_data()[..., t])


def test_compute_rate_of_change_of_chisq():
    """
    Tests compute_rate_of_change_of_chisq function from spm_realign module.
    The said function computes the coefficient matrix "A", used in the
    Newton-Gauss iterated LSP used in the the realigment algo.

    """

    # setup
    true_A = np.array(
        [[-0., -0., -0., -0., -0.,
         -0.],
       [1., -0., -0., -0., 1.000001,
          1.000001],
       [1., -0., -0., -0., 1.0000015,
          1.0000015],
       [-2., 1., -0., 1.000001, -2.000001,
         -5.],
       [1., -0., -0., -0., 1.000001,
          2.000001],
       [1., -0., -0., -0., 1.0000015,
          2.0000015],
       [-2., 1., -0., 1.0000015, -2.000001,
         -6.9999995],
       [1., -0., -0., -0., 1.000001,
          3.000001],
       [1., -0., -0., -0., 1.0000015,
          3.0000015],
       [-2., 1., -0., 1.000002, -2.000001,
         -8.999999],
       [1., -0., -0., -0., 1.000001,
          4.000001],
       [1., -0., -0., -0., 1.0000015,
          4.0000015],
       [-2., -3., 1., -7.0000005, -5.,
          0.9999975],
       [1., -0., -0., -0., 2.000001,
          1.000001],
       [1., -0., -0., -0., 2.0000015,
          1.0000015],
       [-2., 1., -0., 2.000001, -4.000001,
         -5.],
       [1., -0., -0., -0., 2.000001,
          2.000001],
       [1., -0., -0., -0., 2.0000015,
          2.0000015],
       [-2., 1., -0., 2.0000015, -4.000001,
         -6.9999995],
       [1., -0., -0., -0., 2.000001,
          3.000001],
       [1., -0., -0., -0., 2.0000015,
          3.0000015],
       [-2., 1., -0., 2.000002, -4.000001,
         -8.999999],
       [1., -0., -0., -0., 2.000001,
          4.000001],
       [1., -0., -0., -0., 2.0000015,
          4.0000015],
       [-2., -3., 1., -10., -6.9999995,
          0.9999975],
       [1., -0., -0., -0., 3.000001,
          1.000001],
       [1., -0., -0., -0., 3.0000015,
          1.0000015],
       [-2., 1., -0., 3.000001, -6.000001,
         -5.],
       [1., -0., -0., -0., 3.000001,
          2.000001],
       [1., -0., -0., -0., 3.0000015,
          2.0000015],
       [-2., 1., -0., 3.0000015, -6.000001,
         -6.9999995],
       [1., -0., -0., -0., 3.000001,
          3.000001],
       [1., -0., -0., -0., 3.0000015,
          3.0000015],
       [-2., 1., -0., 3.000002, -6.000001,
         -8.999999],
       [1., -0., -0., -0., 3.000001,
          4.000001],
       [1., -0., -0., -0., 3.0000015,
          4.0000015],
       [-2., -3., 1., -12.9999995, -8.999999,
          0.9999975],
       [1., -0., -0., -0., 4.000001,
          1.000001],
       [1., -0., -0., -0., 4.0000015,
          1.0000015],
       [-2., 1., -0., 4.000001, -8.000001,
         -5.],
       [1., -0., -0., -0., 4.000001,
          2.000001],
       [1., -0., -0., -0., 4.0000015,
          2.0000015],
       [-2., 1., -0., 4.0000015, -8.000001,
         -6.9999995],
       [1., -0., -0., -0., 4.000001,
          3.000001],
       [1., -0., -0., -0., 4.0000015,
          3.0000015],
       [-2., 1., -0., 4.000002, -8.000001,
         -8.999999],
       [1., -0., -0., -0., 4.000001,
          4.000001],
       [1., -0., -0., -0., 4.0000015,
          4.0000015],
       [-2., -3., 1., -15.999999, -10.9999985,
          0.9999975],
       [1., -0., -0., -0., 5.000001,
          1.000001],
       [1., -0., -0., -0., 5.0000015,
          1.0000015],
       [-2., 1., -0., 5.000001, -10.000001,
         -5.],
       [1., -0., -0., -0., 5.000001,
          2.000001],
       [1., -0., -0., -0., 5.0000015,
          2.0000015],
       [-2., 1., -0., 5.0000015, -10.000001,
         -6.9999995],
       [1., -0., -0., -0., 5.000001,
          3.000001],
       [1., -0., -0., -0., 5.0000015,
          3.0000015],
       [-2., 1., -0., 5.000002, -10.000001,
         -8.999999],
       [1., -0., -0., -0., 5.000001,
          4.000001],
       [1., -0., -0., -0., 5.0000015,
          4.0000015]])
    decimal_precision = 8  # precision for array comparison
    lkp = [0, 1, 2, 3, 4, 5]  # translations + rotations model
    grid = np.mgrid[1:4:, 1:5:, 1:6:].reshape((3, -1),
                                              # true_A came from matlab
                                              order='F')
    gradG = np.vstack(([0, 0, 0], np.diff(grid, axis=1).T)).T  # image gradient
    M = np.eye(4)  # grid transformation matrix

    # compute A
    A = compute_rate_of_change_of_chisq(M, grid, gradG, lkp=lkp)

    # compare A with true_A (from spm implementation)
    np.testing.assert_array_almost_equal(A, true_A, decimal=decimal_precision)
