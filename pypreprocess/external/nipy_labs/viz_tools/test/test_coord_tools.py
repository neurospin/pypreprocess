# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import numpy.testing
import nibabel

from ..coord_tools import (coord_transform,
                           find_cut_coords,
                           find_maxsep_cut_coords,
                           _maximally_separated_subset
                           )


def test_coord_transform_trivial():
    sform = np.eye(4)
    x = np.random.random((10,))
    y = np.random.random((10,))
    z = np.random.random((10,))

    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x, x_)
    np.testing.assert_array_equal(y, y_)
    np.testing.assert_array_equal(z, z_)

    sform[:, -1] = 1
    x_, y_, z_ = coord_transform(x, y, z, sform)
    np.testing.assert_array_equal(x+1, x_)
    np.testing.assert_array_equal(y+1, y_)
    np.testing.assert_array_equal(z+1, z_)


def test_find_cut_coords():
    map = np.zeros((100, 100, 100))
    x_map, y_map, z_map = 50, 10, 40
    map[x_map-30:x_map+30, y_map-3:y_map+3, z_map-10:z_map+10] = 1
    x, y, z = find_cut_coords(map, mask=np.ones(map.shape, np.bool))
    np.testing.assert_array_equal(
                        (int(round(x)), int(round(y)), int(round(z))),
                                (x_map, y_map, z_map))


def test_maximally_separated_subset():
    numpy.testing.assert_array_equal(_maximally_separated_subset(
            [1, 2, 6, 10], 3), [1, 6, 10])


def test_find_maxsep_cut_coords():
    # simulate fake activation map
    map3d_shape = (18, 19, 23)
    map3d = nibabel.Nifti1Image(np.arange(np.prod(map3d_shape)).reshape
                                (map3d_shape),
                                np.eye(4))

    # slicer == 'x'
    cut_coords = find_maxsep_cut_coords(map3d.get_data(), map3d.get_affine(),
                                        threshold=2.3, slicer='x'
                                        )
    numpy.testing.assert_array_equal(cut_coords, [1, 5, 9, 13, 17])

    # slicer == 'y'
    cut_coords = find_maxsep_cut_coords(map3d.get_data(), map3d.get_affine(),
                                        threshold=2.3, cut_coords=4, slicer='y'
                                        )
    numpy.testing.assert_array_equal(cut_coords, [1, 9, 13, 18])

    # slicer == 'z'
    cut_coords = find_maxsep_cut_coords(map3d.get_data(), map3d.get_affine(),
                                        threshold=2.3, cut_coords=6, slicer='z'
                                        )
    numpy.testing.assert_array_equal(
        cut_coords, [3., 7., 11., 15., 19., 22.])
