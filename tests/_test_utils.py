import nibabel
import numpy as np


def create_random_image(shape=None,
                        ndim=3,
                        n_scans=None,
                        affine=np.eye(4),
                        parent_class=nibabel.Nifti1Image):
    """
    Creates a random image of prescribed shape

    """

    rng = np.random.RandomState(0)

    if shape is None:
        shape = np.random.random_integers(20, size=ndim)

    ndim = len(shape)

    ndim = len(shape)
    if not n_scans is None and ndim == 4:
        shape[-1] = n_scans

    return parent_class(np.random.randn(*shape), affine)
