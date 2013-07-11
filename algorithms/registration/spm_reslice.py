"""
:Module: spm_reslice
:Synopsis: Routine functions for reslicing volumes after affine registration
(as in motion correction, coregistration, etc)
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
import scipy.ndimage
import scipy.linalg
import nibabel
import affine_transformations



def _get_mask(M, coords, dim, wrp=[1, 1, 0], tiny=5e-2):
    """
    Wrapper for get_physical_coords(...) with optional wrapping of dimensions.

    Parameters
    ----------
    M: 2D array of shape (4, 4)
        affine transformation describing voxel-to-world mapping
    coords: array_like of shape (3, n_voxels)
        voxel(s) under consideration
    dim: list of 3 ints
        dimensions (nx, ny, nz) of the voxel space (for example [64, 56, 21])
    wrp: list of 3 bools, optional (default [1, 1, 0])
        each coordinate value indicates whether wrapping should be done in the
        corresponding dimension or not. Possible values are:
        [0, 0, 0]: no wrapping; use this value for PET data
        [1, 1, 0]: wrap all except z (slice-wise) dimension; use this value for
        fMRI data
    tiny: float, optional (default 5e-2)
        threshold for filtering voxels that have fallen out of the FOV

    Returns
    -------
    Tuple (fov_mask, physical_coords), where:
    fov_mask: 1D array_like of len voxel.shape[1]
        mask for filtering voxels that are still in the FOV. 1 means 'OK',
        0 means 'fell out of FOV'
    physical_coords: array of same shape as input coords
        transformed coords

    """

    physical_coords = affine_transformations.get_physical_coords(M, coords)
    fov_mask = np.ones(physical_coords.shape[-1]).astype('bool')

    for j in xrange(3):
        if not wrp[j]:
            fov_mask = fov_mask & (physical_coords[j] >= -tiny
                                   ) & (physical_coords[j] < dim[j] + tiny)

    return fov_mask, physical_coords


def reslice_vols(vols, interp_order=3, interp_mode='wrap',
                 mask=False, wrp=[1, 1, 0], log=None):
    """
    Uses B-spline interpolation to reslice (i.e resample) all other
    volumes to have thesame affine header matrix as the first (0th) volume.

    Parameters
    ----------
    vols: list of `nibabel.Nifti1Image` objects
        vols[0] is the reference volume. All other volumes will be resliced
        so that the end up with the same header affine matrix as vol[0]
    interp_order: int, optional (default 3)
        degree of B-spline interpolation used for resampling the volumes
    interp_mode: string, optional (default "wrap")
        mode param to be passed to `scipy.ndimage.map_coordinates`
    log: function(basestring), optional (default None)
        function for logging messages

    Returns
    -------
    vols: list of `nibabel.Nifti1Image` objects
        resliced volumes

    Raises
    ------
    RuntimeError in case dimensions are inconsistent across volumes.

    """

    def _log(msg):
        """
        Logs given message (msg).

        """

        if log:
            log(msg)
        else:
            print(msg)

    # build working grid
    dim = vols[0].shape
    grid = np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]].reshape((3, -1))

    # compute global mask for all vols, to mask out voxels that show
    # artefactual movement across volumes
    msk = np.ones(grid.shape[1]).astype('bool')
    for t in xrange(1, len(vols)):
        # saniiy check on dimensions
        if vols[t].shape != dim:
            raise RuntimeError(
                ("All source volumes must have the same dimensions as the "
                 "reference. Volume %i has dim %s instead of %s.") % (
                    t, vols[t].shape, dim))

        # affine matrix for passing from vol's space to the ref vol's
        M = scipy.linalg.inv(scipy.linalg.lstsq(
                vols[0].get_affine(), vols[t].get_affine())[0])

        fov_msk, _ = _get_mask(M, grid, dim, wrp=wrp)

        msk = msk & fov_msk

    # loop on all vols --except the ref vol-- reslicing them one-by-one
    for t in xrange(len(vols)):
        _log('\tReslicing volume %i/%i...' % (t + 1, len(vols)))

        if t > 0:
            # affine matrix for passing from vol's space to the ref vol's
            M = scipy.linalg.inv(scipy.linalg.lstsq(
                    vols[0].get_affine(), vols[t].get_affine())[0])

            # transform vol's grid according to M
            _, new_grid = _get_mask(M, grid, dim, wrp=wrp)

            # resample vol on new grid
            rdata = scipy.ndimage.map_coordinates(
                vols[t].get_data(), new_grid,
                order=interp_order,
                mode=interp_mode,
                )
        else:
            # don't reslice reference vol
            rdata = vols[t].get_data().ravel()

        # mask out voxels that have fallen out of the global mask
        rdata[~msk] = 0  # XXX should really be set to NaN

        # replace vols's affine with ref vol's (this has been the ultimate
        # goal all along)
        vols[t] = nibabel.Nifti1Image(rdata.reshape(dim),
                                      vols[0].get_affine())

    # returned resliced vols
    return vols
