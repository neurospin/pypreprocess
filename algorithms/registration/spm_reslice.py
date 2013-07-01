"""
:Module: spm_reslice
:Synopsis: Routine functions for reslicing volumes post-registration
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
import scipy.ndimage
import scipy.linalg
import nibabel
import affine_transformations


def reslice_vols(vols, interp=3, log=None):
    """
    Uses B-spline interpolation to reslice (i.e resample) all other
    volumes to have thesame affine header matrix as the first (0th) volume.

    Parameters
    ----------
    vols: list of `nibabel.Nifti1Image` objects
        vols[0] is the reference volume. All other volumes will be resliced
        so that the end up with the same header affine matrix as vol[0]
    interp: int, optional (default 3)
        degree of B-spline interpolation used for resampling the volumes

    Returns
    -------
    vols: list of `nibabel.Nifti1Image` objects
        resliced volumes

    Raises
    ------
    RuntimeError

    """

    def _log(msg):
        if log:
            log(msg)
        else:
            print(msg)

    # build working grid
    dim = vols[0].shape
    grid = np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]].reshape((3, -1))

    # loop on all vols --except the ref vol-- reslicing them one-by-one
    for t in xrange(len(vols)):
        if t == 0:
            continue

        _log('\tReslicing volume %i/%i...' % (t + 1, len(vols)))

        # sanitiy check on dimensions
        if vols[t].shape != dim:
            raise RuntimeError(
                ("All source volumes must have the same dimensions as the "
                 "reference. Volume %i has dim %s instead of %s.") % (
                    t, vols[t].shape, dim))

        # affine matrix for passing from vol's space to the ref vol's
        M = scipy.linalg.inv(scipy.linalg.lstsq(
                vols[0].get_affine(), vols[t].get_affine())[0])

        # transform vol's grid according to M
        fov_mask, new_grid = affine_transformations.get_mask(
            M, grid, dim, wrp=[0, 0, 0])
        print new_grid

        # resample vol on transformed grid
        rdata = scipy.ndimage.map_coordinates(vols[t].get_data(), new_grid,
                                              order=interp,
                                              mode='wrap')

        # mask out voxels fallen out of FOV
        # rdata[fov_mask] = 0

        # replace vols's affine with ref vol's (this was the
        # goal all along)
        vols[t] = nibabel.Nifti1Image(rdata.reshape(dim),
                                      vols[0].get_affine())

    # returned resliced vols
    return vols
