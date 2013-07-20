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


def _load_vol(x):
    """
    Loads a single 3D volume.

    """

    if isinstance(x, basestring):
        vol = nibabel.load(x)
    elif isinstance(x, nibabel.Nifti1Image) or isinstance(
        x, nibabel.Nifti1Pair):
        vol = x
    else:
        raise TypeError(
            ("Each volume must be string, image object, got:"
             " %s") % type(x))

    if len(vol.shape) == 4:
        if vol.shape[-1] == 1:
            vol = nibabel.Nifti1Image(vol.get_data()[..., 0],
                                      vol.get_affine())
        else:
            raise ValueError(
                "Each volume must be 3D, got %iD" % len(vol.shape))
    elif len(vol.shape) != 3:
            raise ValueError(
                "Each volume must be 3D, got %iD" % len(vol.shape))

    return vol


def reslice_vols(vols, target_affine=None, interp_order=3,
                 interp_mode='constant', mask=True, wrp=[1, 1, 0], log=None):
    """
    Uses B-spline interpolation to reslice (i.e resample) all other
    volumes to have thesame affine header matrix as the first (0th) volume.

    Parameters
    ----------
    vols: list of `nibabel.Nifti1Image` objects
        vols[0] is the reference volume. All other volumes will be resliced
        so that the end up with the same header affine matrix as vol[0]
    target_affine: 2D array of shape (4, 4), optional (default None)
        target affine matrix to which the vols will be resliced. If not
        specified, vols will be resliced to match the first vol's affine
    interp_order: int, optional (default 3)
        degree of B-spline interpolation used for resampling the volumes
    interp_mode: string, optional (default "wrap")
        mode param to be passed to `scipy.ndimage.map_coordinates`
    mask: boolean, optional (default True)
        if set, vols will be masked before reslicing. This masking will
        help eliminate artefactual motion across volumes due to on-off
        voxels
    wrp: list_like of 3 booleans, optional (default [1, 1, 0])
        option passed to _get_mask function. For each axis, it specifies
        if or not wrapping is to be done along that axis
    log: function(basestring), optional (default None)
        function for logging messages

    Returns
    -------
    vols: generator object on `nibabel.Nifti1Image` objects
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

    vols = list(vols)

    # load first vol
    vol_0 = _load_vol(vols[0])

    # sanitize target_affine
    reslice_first_vol = True
    if target_affine is None:
        reslice_first_vol = False
        target_affine = vol_0.get_affine()

    # build working grid
    dim = vol_0.shape
    n_scans = len(vols)
    grid = np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]].reshape((3, -1))

    # compute global mask for all vols, to mask out voxels that show
    # artefactual movement across volumes
    msk = np.ones(grid.shape[1]).astype('bool')
    if mask:
        for t in xrange(len(vols)):
            # load vol
            vol = _load_vol(vols[t])

            # saniiy check on dimensions
            if vol.shape != dim:
                raise RuntimeError(
                    ("All source volumes must have the same dimensions as the "
                     "reference. Volume %i has dim %s instead of %s.") % (
                        t, vol.shape, dim))

            # affine matrix for passing from vol's space to the ref vol's
            M = scipy.linalg.inv(scipy.linalg.lstsq(
                    target_affine, vol.get_affine())[0])

            fov_msk, _ = _get_mask(M, grid, dim, wrp=wrp)

            msk = msk & fov_msk

    # loop on all vols, reslicing them one-by-one
    for t in xrange(n_scans):
        _log('\tReslicing volume %i/%i...' % (t + 1, len(vols)))

        # load vol
        vol = _load_vol(vols[t])

        # reslice vol
        if t > 0 or reslice_first_vol:
            # affine matrix for passing from vol's space to the ref vol's
            M = scipy.linalg.inv(scipy.linalg.lstsq(
                    target_affine,
                    vol.get_affine())[0])

            # transform vol's grid according to M
            _, new_grid = _get_mask(M, grid, dim, wrp=wrp)

            # resample vol on new grid
            rdata = scipy.ndimage.map_coordinates(
                vol.get_data(), new_grid,
                order=interp_order,
                mode=interp_mode
                )
        else:  # don't reslice first vol
            rdata = vol.get_data().ravel()

        rdata[~msk]  = 0

        # replace vols's affine with ref vol's (this has been the ultimate
        # goal all along)
        rvol = nibabel.Nifti1Image(rdata.reshape(dim), target_affine)

        # yield resliced vol
        yield rvol
