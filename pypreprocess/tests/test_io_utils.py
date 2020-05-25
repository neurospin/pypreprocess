import os
import tempfile
import inspect
import numpy as np
import nibabel
from nilearn.image.image import check_niimg_4d
from numpy.testing import assert_array_equal

from pypreprocess.io_utils import delete_orientation

from ..io_utils import (
    do_3Dto4D_merge, load_vols, save_vols, save_vol, hard_link, nii2niigz,
    get_basename, get_basenames, is_niimg, is_4D, is_3D, get_vox_dims,
    niigz2nii, _expand_path, isdicom, get_shape, get_relative_path,
    loaduint8)

# global setup
this_file = os.path.basename(os.path.abspath(__file__)).split('.')[0]
OUTPUT_DIR = "/tmp/%s" % this_file
IMAGE_EXTENSIONS = [".nii", ".nii.gz", ".img"]


def create_random_image(shape=None, ndim=3, n_scans=None, affine=np.eye(4),
                        parent_class=nibabel.Nifti1Image):
    """
    Creates a random image of prescribed shape

    """
    if not n_scans is None:
        ndim = 4
    if shape is None:
        shape = np.random.random_integers(20, size=ndim)
    ndim = len(shape)
    if not n_scans is None and ndim == 4:
        shape[-1] = n_scans
    return parent_class(np.random.randn(*shape), affine)


def test_save_vol():
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vol = create_random_image(ndim=3)
    output_filename = save_vol(vol, output_dir=output_dir,
                               basename='123.nii.gz')
    assert os.path.basename(output_filename) == '123.nii.gz'
    output_filename = save_vol(vol, output_dir=output_dir, basename='123.img',
                               prefix='s')
    assert os.path.basename(output_filename) == 's123.img'


def test_save_vols():
    # setup
    n_scans = 10
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create 4D film
    film = create_random_image(ndim=4, n_scans=n_scans)
    threeD_vols = nibabel.four_to_three(film)

    # save vols manually
    film_filename = os.path.join(output_dir, 'film.nii.gz')
    threeD_vols_filenames = [os.path.join(output_dir, 'fMETHODS-%06i' % i)
                             for i in range(len(threeD_vols))]

    # check saving seperate 3D vols
    for stuff in [film, threeD_vols]:
        if isinstance(stuff, list):
            basenames = [os.path.basename(x)
                         for x in threeD_vols_filenames]
        else:
            basenames = os.path.basename(film_filename)
        for concat in [False, True]:
            for bn in [None, basenames]:
                saved_vols_filenames = save_vols(stuff,
                                                 output_dir,
                                                 ext='.nii.gz',
                                                 concat=concat,
                                                 basenames=bn
                                                 )
                if not concat and isinstance(stuff, list):
                    assert isinstance(saved_vols_filenames, list)
                    assert len(saved_vols_filenames) == n_scans
                    if not bn is None:
                        assert os.path.basename(saved_vols_filenames[7]) == 'fMETHODS-000007.nii.gz'
                else:
                    assert isinstance(saved_vols_filenames, str)
                    assert saved_vols_filenames.endswith('.nii.gz'), saved_vols_filenames
                    assert is_4D(check_niimg_4d(saved_vols_filenames))


def test_save_vols_from_ndarray_with_affine():
    # setup
    n_scans = 10
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create 4D film
    film = np.random.randn(5, 7, 11, n_scans)
    threeD_vols = [film[..., t] for t in range(n_scans)]

    # check saving seperate 3D vols
    for stuff in [film, threeD_vols]:
        for concat in [False, True]:
            saved_vols_filenames = save_vols(
                stuff, output_dir, ext='.nii.gz', affine=np.eye(4),
                concat=concat)
            if not concat and isinstance(stuff, list):
                assert isinstance(saved_vols_filenames, list)
                assert len(saved_vols_filenames) == n_scans
            else:
                assert isinstance(saved_vols_filenames, str)


def test_do_3Dto4D_merge():
    # setup
    n_scans = 10
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create 4D film
    film = create_random_image(ndim=4, n_scans=n_scans)
    threeD_vols = nibabel.four_to_three(film)

    _film = do_3Dto4D_merge(threeD_vols)

    assert _film.shape == film.shape

    save_vols(threeD_vols, output_dir, ext='.nii.gz')


def test_hardlink():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def _touch(filename):
        with open(filename, 'a') as fd:
            fd.close()

    def _make_filenames(n=1):
        """
        Helper function to make a filename or list of filenames,
        or list of lists of such, or ...

        """

        if n < 2 or np.random.rand() < .5:
            filename = tempfile.mktemp()

            # file extension
            extensions = [".nii.gz"] if np.random.rand() < .5 else [
                ".img", ".hdr"]

            files = [filename + ext for ext in extensions]

            for x in files:
                _touch(x)

            return files[0]
        else:
            l = np.random.randint(1, n)
            return [_make_filenames(n - l) for _ in range(l)]

    filenames = _make_filenames()
    hl_filenames = hard_link(filenames, output_dir)

    def _check_ok(x, y):
        if isinstance(x, str):
            # check that hardlink was actually made
            assert os.path.exists(x)
            if x.endswith('.img'):
                assert os.path.exists(x.replace(".img", ".hdr"))

            # cleanup
            os.unlink(x)
            os.remove(y)
            if x.endswith('.img'):
                os.unlink(x.replace(".img", ".hdr"))
                os.unlink(y.replace(".img", ".hdr"))
        else:
            # assuming list_like; recursely do this check
            assert isinstance(x, list)
            assert isinstance(y, list)

            for _x, _y in zip(x, y):
                _check_ok(_x, _y)

    _check_ok(hl_filenames, filenames)


def test_get_basename():
    assert get_basename("/tmp/toto/titi.nii.gz", ext=".img") == "titi.img"
    assert get_basename("/tmp/toto/titi.nii.gz") == "titi.nii.gz"
    assert get_basename("/tmp/toto/titi.nii", ext=".img") == "titi.img"
    assert get_basename("/tmp/toto/titi.nii") ==  "titi.nii"


def test_get_basenames():
    assert get_basenames("/path/to/file/file.nii.gz") == "file.nii.gz"
    assert get_basenames(["/path/to/file/file-%04i.nii.gz" % i
                                for i in range(10)])[3] ==  "file-0003.nii.gz"


def test_get_vox_dims():
    # setup
    affine = np.eye(4)
    np.fill_diagonal(affine, [-3, 3, 3])

    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3D vol
    vol = create_random_image(affine=affine)
    np.testing.assert_array_equal(get_vox_dims(vol), [3, 3, 3])

    # 3D image file
    saved_img_filename = os.path.join(output_dir, "vol.nii.gz")
    nibabel.save(vol, saved_img_filename)
    np.testing.assert_array_equal(get_vox_dims(vol), [3, 3, 3])

    # 4D niimg
    film = create_random_image(n_scans=10, affine=affine)
    np.testing.assert_array_equal(get_vox_dims(film), [3, 3, 3])

    # 4D image file
    film = create_random_image(n_scans=10, affine=affine)
    saved_img_filename = os.path.join(output_dir, "4D.nii.gz")
    nibabel.save(film, saved_img_filename)
    np.testing.assert_array_equal(get_vox_dims(film), [3, 3, 3])


def test_is_niimg():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4D niimg
    film = create_random_image(n_scans=10)
    assert is_niimg(film)

    # 3D niimg
    vol = create_random_image()
    assert is_niimg(vol)

    # filename is not niimg
    assert not is_niimg("/path/to/some/nii.gz")


def test_niigz2nii_with_filename():
    # create and save .nii.gz image
    img = create_random_image()
    ifilename = '/tmp/toto.nii.gz'
    nibabel.save(img, ifilename)

    # convert img to .nii
    ofilename = niigz2nii(ifilename, output_dir='/tmp/titi')

    # checks
    assert ofilename == '/tmp/titi/toto.nii'
    nibabel.load(ofilename)


def test_niigz2nii_with_list_of_filenames():
    # creates and save .nii.gz image
    ifilenames = []
    for i in range(4):
        img = create_random_image()
        ifilename = '/tmp/img%i.nii.gz' % i
        nibabel.save(img, ifilename)
        ifilenames.append(ifilename)

    # convert imgs to .nii
    ofilenames = niigz2nii(ifilenames, output_dir='/tmp/titi')

    # checks
    assert len(ifilenames) == len(ofilenames)
    for x in range(len(ifilenames)):
        nibabel.load(ofilenames[x])


def test_niigz2nii_with_list_of_lists_of_filenames():
    # creates and save .nii.gz image
    ifilenames = []
    for i in range(4):
        img = create_random_image()
        ifilename = '/tmp/img%i.nii.gz' % i
        nibabel.save(img, ifilename)
        ifilenames.append(ifilename)

    # convert imgs to .nii
    ofilenames = niigz2nii([ifilenames], output_dir='/tmp/titi')

    # checks
    assert len(ofilenames) == 1
    for x in range(len(ofilenames[0])):
        nibabel.load(ofilenames[0][x])


def test_expand_path():
    # paths with . (current directory)
    assert _expand_path("./my/funky/brakes", relative_to="/tmp") == "/tmp/my/funky/brakes"

    # paths with .. (parent directory)
    assert _expand_path("../my/funky/brakes", relative_to="/tmp") == "/my/funky/brakes"
    assert _expand_path(".../my/funky/brakes", relative_to="/tmp") == None

    # paths with tilde
    assert _expand_path("~/my/funky/brakes") == os.path.join(os.environ['HOME'], "my/funky/brakes")
    assert _expand_path("my/funky/brakes", relative_to="~") == os.path.join(os.environ['HOME'], "my/funky/brakes")


def test_isdicom():
    # +ve
    assert isdicom("/toto/titi.dcm")
    assert isdicom("/toto/titi.DCM")
    assert isdicom("/toto/titi.ima")
    assert isdicom("/toto/titi.IMA")

    # -ve
    assert not isdicom("/toto/titi.nii.gz")
    assert not isdicom("/toto/titi.nii")
    assert not isdicom("/toto/titi.img")
    assert not isdicom("/toto/titi.hdr")
    assert not isdicom("bad")


def test_is_3D():
    vol = create_random_image(ndim=3)
    assert is_3D(vol)
    assert not is_4D(vol)
    assert not is_3D(create_random_image(shape=(64, 64, 64, 1)))


def test_is_4d():
    film = create_random_image(ndim=4)
    assert is_4D(film)
    assert not is_3D(film)
    assert is_4D(create_random_image(shape=(64, 64, 64, 1)))


def test_get_shape():
    shape = (61, 62, 63, 64)
    img = create_random_image(shape)
    assert get_shape(img) == shape

    shape = (34, 45, 65)
    n_scans = 10
    img = [create_random_image(shape) for _ in range(n_scans)]
    assert get_shape(img) == tuple(list(shape) + [n_scans])


def test_get_relative_path():
    assert get_relative_path("dop/", "dop/rob") == "rob"

    assert get_relative_path("/toto/titi", "/toto/titi/tata/test.txt") == "tata/test.txt"
    assert get_relative_path("/toto/titi", "/toto/titi/tata/") == "tata"
    assert get_relative_path("/toto/titi", "/toto/titI/tato/dada") == None
    assert get_relative_path("/toto/titi", "/toto/titi") == ""


def test_load_vols():
    vol = nibabel.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    assert len(load_vols([vol])) == 1

    vols = load_vols(vol)

    # all loaded vols should be 3-dimensional
    for v in vols:
        assert len(v.shape) == 3

    assert len(vols) == 1


def test_loaduint8():
    # bullet-proof for a certain crash
    vol = nibabel.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
    loaduint8(vol)


def test_load_vols_different_affine():
    # check_niimg_4d crashes when loading list of 3d vols with different
    # affines
    i1 = np.eye(4)
    i2 = np.eye(4)
    i2[:-1, -1] = 5.
    vol1 = nibabel.Nifti1Image(np.zeros((3, 3, 3)), i1)
    vol2 = nibabel.Nifti1Image(np.zeros((3, 3, 3)), i2)
    load_vols([vol1, vol2])


def test_load_vols_from_single_filename():
    for flim in [True, False][0:]:
        shape = [2, 3, 4]
        if flim: shape += [5]
        vols = nibabel.Nifti1Image(np.zeros(shape), np.eye(4))
        vols.to_filename("/tmp/test.nii.gz")
        vols = load_vols("/tmp/test.nii.gz")
        for vol in vols: assert vol.shape == tuple(shape[:3])


def test_load_vols_from_singleton_list_of_4D_img():
    shape = [2, 3, 4]
    for n_scans in [1, 5]:
        for from_file in [True, False]:
            vols = nibabel.Nifti1Image(np.zeros(shape + [n_scans]), np.eye(4))
            if from_file:
                vols.to_filename("/tmp/test.nii.gz")
                vols = "/tmp/test.nii.gz"
            vols = load_vols([vols])
            assert len(vols) == n_scans


def test_delete_orientation():
    i1 = np.eye(4)
    i1[:-1, -1] = 5.
    vol1 = nibabel.Nifti1Image(np.zeros((3, 3, 3)), i1)
    nibabel.save(vol1, '/tmp/vol1.nii.gz')
    vol1 = '/tmp/vol1.nii.gz'
    delete_orientation(vol1, '/tmp', output_tag='del_')
    vol1 = nibabel.load('/tmp/vol1.nii.gz')
    vol2 = nibabel.load('/tmp/del_vol1.nii.gz')
    data_vol1 = vol1.get_data()
    data_vol2 = vol2.get_data()
    assert_array_equal(data_vol1, data_vol2)
    header = vol2.get_header()
    for key in ['dim_info', 'quatern_b', 'quatern_c', 'quatern_d',
                'qoffset_x', 'qoffset_y', 'qoffset_z',
                'srow_x', 'srow_x', 'srow_z']:
        print(header[key])
        assert_array_equal(header[key], 0)


def test_nii2niigz_with_filename():
    # create and save .nii image
    img = create_random_image()
    ifilename = '/tmp/toto.nii'
    nibabel.save(img, ifilename)

    # convert img to .nii.gz
    ofilename = nii2niigz(ifilename, output_dir='/tmp/titi')

    # checks
    assert ofilename == '/tmp/titi/toto.nii.gz'
    nibabel.load(ofilename)


def test_nii2niigz_with_list_of_filenames():
    # creates and save .nii image
    ifilenames = []
    for i in range(4):
        img = create_random_image()
        ifilename = '/tmp/img%i.nii' % i
        nibabel.save(img, ifilename)
        ifilenames.append(ifilename)

    # convert imgs to .nii.gz
    ofilenames = nii2niigz(ifilenames, output_dir='/tmp/titi')

    # checks
    assert len(ifilenames) == len(ofilenames)
    for x in range(len(ifilenames)):
        nibabel.load(ofilenames[x])


def test_nii2niigz_with_list_of_lists_of_filenames():
    # creates and save .nii image
    ifilenames = []
    for i in range(4):
        img = create_random_image()
        ifilename = '/tmp/img%i.nii' % i
        nibabel.save(img, ifilename)
        ifilenames.append(ifilename)

    # convert imgs to .nii.gz
    ofilenames = nii2niigz([ifilenames], output_dir='/tmp/titi')

    # checks
    assert len(ofilenames) == 1
    for x in range(len(ofilenames[0])):
        nibabel.load(ofilenames[0][x])
