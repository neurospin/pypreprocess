"""
:Module: spm_reslice
:Synopsis: Routine functions for reslicing volumes post affine registration
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
import scipy.ndimage
import scipy.linalg
import nibabel
import affine_transformations


def reslice_vols(vols, interp=3, mask=False, wrp=[1, 1, 0], log=None):
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

        fov_msk, new_grid = affine_transformations.get_mask(M, grid, dim,
                                                            wrp=wrp)

        msk = msk & fov_msk

    # loop on all vols --except the ref vol-- reslicing them one-by-one
    for t in xrange(len(vols)):
        _log('\tReslicing volume %i/%i...' % (t + 1, len(vols)))

        if t > 0:
            # affine matrix for passing from vol's space to the ref vol's
            M = scipy.linalg.inv(scipy.linalg.lstsq(
                    vols[0].get_affine(), vols[t].get_affine())[0])

            # transform vol's grid according to M
            _, new_grid = affine_transformations.get_mask(M, grid, dim,
                                                                wrp=wrp)

            # resample vol on transformed grid
            rdata = scipy.ndimage.map_coordinates(
                vols[t].get_data(), new_grid,
                order=interp,

                # wrapping, reflecting, etc., at the boundaries may produce
                # artefactual gray values that will cause artefactual motion
                # across volumes; setting values at the boundaries and to zero
                # will help avoid this
                mode="constant",
                cval=0.
                )
        else:
            # don't reslice reference vol
            rdata = vols[t].get_data()

        # mask out voxels that have fallen out of the global mask
        rdata[~msk] = 0  # XXX should really be set to NaN

        # replace vols's affine with ref vol's (this has been the ultimate
        # goal all along)
        vols[t] = nibabel.Nifti1Image(rdata.reshape(dim),
                                      vols[0].get_affine())

    # returned resliced vols
    return vols
