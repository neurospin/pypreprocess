import os
import numpy as np
import pytest
import numpy.testing
import nibabel
import scipy.io
from ..coreg import compute_similarity_from_jhist, Coregister
from ..affine_transformations import apply_realignment_to_vol
from .test_histograms import test_joint_histogram

# global setup
THIS_FILE = os.path.abspath(__file__).split('.')[0]
THIS_DIR = os.path.dirname(THIS_FILE)
OUTPUT_DIR = "/tmp/%s" % os.path.basename(THIS_FILE)


def test_compute_similarity_from_jhist():
    jh = test_joint_histogram()

    for cost_fun in ['mi', 'nmi', 'ecc']:
        s = compute_similarity_from_jhist(jh, cost_fun=cost_fun)
        assert not (s > 1)

@pytest.mark.skip()
def test_coregister_on_toy_data():
    shape = (23, 29, 31)
    ref = nibabel.Nifti1Image(np.arange(np.prod(shape)).reshape(shape),
                              np.eye(4)
                              )

    # rigidly move reference vol to get a new volume: the source vol
    src = apply_realignment_to_vol(ref, [1, 1, 1,  # translations
                                         0, .01, 0,  # rotations
                                         ])

    # learn realignment params for coregistration: src -> ref
    c = Coregister(sep=[4, 2, 1]).fit(ref, src)

    # compare estimated realigment parameters with ground-truth
    numpy.testing.assert_almost_equal(-c.params_[4], .01, decimal=2)
    numpy.testing.assert_array_almost_equal(-c.params_[[3, 5]],
                                             [0, 0], decimal=2)
    numpy.testing.assert_array_equal(np.round(-c.params_)[[0, 1, 2]],
                                     [1., 1., 1.])


@pytest.mark.skip()
def test_coregister_on_real_data():
    # load data
    _tmp = scipy.io.loadmat(
        os.path.join(THIS_DIR, "test_data/some_anat.mat"),
        squeeze_me=True, struct_as_record=False)
    ref = nibabel.Nifti1Image(_tmp['data'], _tmp['affine'])

    # rigidly move reference vol to get a new volume: the source vol
    src = apply_realignment_to_vol(ref, [1, 2, 3,  # translations
                                         0, .01, 0,  # rotations
                                         ])

    # learn realignment params for coregistration: src -> ref
    c = Coregister().fit(ref, src)

    # compare estimated realigment parameters with ground-truth
    numpy.testing.assert_almost_equal(-c.params_[4], .01, decimal=4)
    numpy.testing.assert_array_almost_equal(-c.params_[[3, 5]],
                                             [0, 0], decimal=4)
    numpy.testing.assert_array_equal(np.round(-c.params_)[[0, 1, 2]],
                                     [1., 2., 3.])
