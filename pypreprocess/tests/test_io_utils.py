import numpy as np
import numpy.testing
import nibabel
import os
import tempfile
import inspect
import nose
import nose.tools

# import the APIIS to be tested
from ..io_utils import (
    load_vol,
    load_specific_vol,
    do_3Dto4D_merge,
    save_vols,
    save_vol,
    hard_link,
    get_basenames,
    load_4D_img,
    is_niimg,
    is_4D,
    get_vox_dims,
    niigz2nii
    )

# global setup
this_file = os.path.basename(os.path.abspath(__file__)).split('.')[0]
OUTPUT_DIR = "/tmp/%s" % this_file
IMAGE_EXTENSIONS = [".nii", ".nii.gz", ".img"]


def create_random_image(shape=None,
                        ndim=3,
                        n_scans=None,
                        affine=np.eye(4),
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


def test_load_vol():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # creat a volume
    vol = create_random_image()

    # test loading vol from nibabel object
    _vol = load_vol(vol)
    nose.tools.assert_true(isinstance(_vol, type(vol)))
    nose.tools.assert_equal(_vol.shape, vol.shape)
    numpy.testing.assert_array_equal(_vol.get_data(), vol.get_data())

    # test loading vol by filename
    for ext in IMAGE_EXTENSIONS:
        # save vol with extension ext
        vol_filename = os.path.join(output_dir, "vol%s" % ext)
        nibabel.save(vol, vol_filename)

        # note that .img loads as Nifti1Pair, not Nifti1Image
        vol_type = nibabel.Nifti1Pair if ext == '.img' else nibabel.Nifti1Image

        # load the vol by filename
        _vol = load_vol(vol_filename)
        nose.tools.assert_true(isinstance(_vol, vol_type))
        nose.tools.assert_equal(_vol.shape, vol.shape)
        numpy.testing.assert_array_equal(_vol.get_data(), vol.get_data())


def test_load_specific_vol():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    n_scans = 23

    # create 4D film
    film = create_random_image(ndim=4, n_scans=n_scans)

    # test loading vol from nibabel image object
    for t in xrange(n_scans):
        _vol, _n_scans = load_specific_vol(film, t)
        nose.tools.assert_equal(_n_scans, n_scans)
        nose.tools.assert_true(isinstance(_vol, type(film)))
        nose.tools.assert_equal(_vol.shape, film.shape[:-1])
        numpy.testing.assert_array_equal(_vol.get_data(),
                                         film.get_data()[..., t])

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
                _vol, _n_scans = load_specific_vol(film_filename, t)
                nose.tools.assert_equal(_n_scans, n_scans)
                nose.tools.assert_true(isinstance(_vol, vol_type))
                nose.tools.assert_equal(_vol.shape, film.shape[:-1])
                numpy.testing.assert_array_equal(_vol.get_data(),
                                              film.get_data()[..., t])


def test_save_vol():
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vol = create_random_image(ndim=3)

    output_filename = save_vol(vol, output_dir=output_dir, basename='123')
    nose.tools.assert_equal(os.path.basename(output_filename),
                            '123.nii.gz')

    output_filename = save_vol(vol, output_dir=output_dir, basename='123',
                                prefix='s')
    nose.tools.assert_equal(os.path.basename(output_filename),
                            's123.nii.gz')


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
                             for i in xrange(len(threeD_vols))]

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
                                                 basenames=basenames
                                                 )
                if not concat and isinstance(stuff, list):
                        nose.tools.assert_true(isinstance(
                                saved_vols_filenames, list))
                        nose.tools.assert_equal(len(saved_vols_filenames),
                                                n_scans)

                        nose.tools.assert_equal(os.path.basename(saved_vols_filenames[7]),
                                                'fMETHODS-000007.nii.gz')
                else:
                    nose.tools.assert_true(isinstance(saved_vols_filenames, basestring))
                    nose.tools.assert_true(saved_vols_filenames.endswith('.nii.gz'))
                    nose.tools.assert_true(is_4D(load_4D_img(
                                saved_vols_filenames)))


def test_save_vols_from_ndarray_with_affine():
    # setup
    n_scans = 10
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create 4D film
    film = np.random.randn(5, 7, 11, n_scans)
    threeD_vols = [film[..., t] for t in xrange(n_scans)]

    # check saving seperate 3D vols
    for stuff in [film, threeD_vols]:
        for concat in [False, True]:
            for affine in [None, np.eye(4)]:
                saved_vols_filenames = save_vols(stuff,
                                                  output_dir,
                                                  ext='.nii.gz',
                                                  affine=np.eye(4),
                                                  concat=concat
                                                  )
                if not concat and isinstance(stuff, list):
                        nose.tools.assert_true(isinstance(
                                saved_vols_filenames, list))
                        nose.tools.assert_equal(len(saved_vols_filenames),
                                                n_scans)
                else:
                    nose.tools.assert_true(isinstance(saved_vols_filenames, basestring))


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

    nose.tools.assert_equal(_film.shape, film.shape)

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
            return [_make_filenames(n - l) for _ in xrange(l)]

    filenames = _make_filenames()
    hl_filenames = hard_link(filenames, output_dir)

    def _check_ok(x, y):
        if isinstance(x, basestring):
            # check that hardlink was actually made
            nose.tools.assert_true(os.path.exists(x))
            if x.endswith('.img'):
                nose.tools.assert_true(os.path.exists(x.replace(".img", ".hdr")))

            # cleanup
            os.unlink(x)
            os.remove(y)
            if x.endswith('.img'):
                os.unlink(x.replace(".img", ".hdr"))
                os.unlink(y.replace(".img", ".hdr"))
        else:
            # assuming list_like; recursely do this check
            nose.tools.assert_true(isinstance(x, list))
            nose.tools.assert_true(isinstance(y, list))

            for _x, _y in zip(x, y):
                _check_ok(_x, _y)

    _check_ok(hl_filenames, filenames)


def test_get_basenames():
    nose.tools.assert_equal(get_basenames("/path/to/file/file.nii.gz"),
                            "file")

    nose.tools.assert_equal(get_basenames(["/path/to/file/file-%04i.nii.gz" % i
                                           for i in xrange(10)])[3],
                            "file-0003")


def test_load_4D_img():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # try loading from 4D niimg
    film = create_random_image(n_scans=10)
    loaded_4D_img = load_4D_img(film)
    nose.tools.assert_true(is_niimg(loaded_4D_img))
    nose.tools.assert_equal(loaded_4D_img.shape, film.shape)

    # try loading from 4D image file
    film = create_random_image(n_scans=10)
    saved_img_filename = os.path.join(output_dir, "4D.nii.gz")
    nibabel.save(film, saved_img_filename)
    loaded_4D_img = load_4D_img(saved_img_filename)
    nose.tools.assert_true(is_niimg(loaded_4D_img))
    nose.tools.assert_equal(loaded_4D_img.shape, film.shape)

    # try loading from list of 3D niimgs
    film = create_random_image(n_scans=10)
    loaded_4D_img = load_4D_img(nibabel.four_to_three(film))
    nose.tools.assert_true(is_niimg(loaded_4D_img))
    nose.tools.assert_equal(loaded_4D_img.shape, film.shape)

    # try loading from list of 3D image files
    film = create_random_image(n_scans=10)
    saved_vols_filenames = save_vols(film,
                                     output_dir,
                                     ext='.nii.gz',
                                     )
    loaded_4D_img = load_4D_img(saved_vols_filenames)
    nose.tools.assert_true(is_niimg(loaded_4D_img))
    nose.tools.assert_equal(loaded_4D_img.shape, film.shape)


def test_get_vox_dims():
    # setup
    affine = np.eye(4)
    np.fill_diagonal(affine, [-3, 3, 3])

    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3D vol
    vol = create_random_image(affine=affine)
    numpy.testing.assert_array_equal(get_vox_dims(vol), [3, 3, 3])

    # 3D image file
    saved_img_filename = os.path.join(output_dir, "vol.nii.gz")
    nibabel.save(vol, saved_img_filename)
    numpy.testing.assert_array_equal(get_vox_dims(vol), [3, 3, 3])

    # 4D niimg
    film = create_random_image(n_scans=10, affine=affine)
    numpy.testing.assert_array_equal(get_vox_dims(film), [3, 3, 3])

    # 4D image file
    film = create_random_image(n_scans=10, affine=affine)
    saved_img_filename = os.path.join(output_dir, "4D.nii.gz")
    nibabel.save(film, saved_img_filename)
    numpy.testing.assert_array_equal(get_vox_dims(film), [3, 3, 3])


def test_is_niimg():
    # setup
    output_dir = os.path.join(OUTPUT_DIR, inspect.stack()[0][3])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4D niimg
    film = create_random_image(n_scans=10)
    nose.tools.assert_true(is_niimg(film))

    # 3D niimg
    vol = create_random_image()
    nose.tools.assert_true(is_niimg(vol))

    # filename is not niimg
    nose.tools.assert_false(is_niimg("/path/to/some/nii.gz"))


def test_niigz2nii_with_filename():
    # create and save .nii.gz image
    img = create_random_image()
    ifilename = '/tmp/toto.nii.gz'
    nibabel.save(img, ifilename)

    # convert img to .nii
    ofilename = niigz2nii(ifilename, output_dir='/tmp/titi')

    # checks
    nose.tools.assert_equal(ofilename, '/tmp/titi/toto.nii')
    nibabel.load(ofilename)


def test_niigz2nii_with_list_of_filenames():
    # creates and save .nii.gz image
    ifilenames = []
    for i in xrange(4):
        img = create_random_image()
        ifilename = '/tmp/img%i.nii.gz' % i
        nibabel.save(img, ifilename)
        ifilenames.append(ifilename)

    # convert imgs to .nii
    ofilenames = niigz2nii(ifilenames, output_dir='/tmp/titi')

    # checks
    nose.tools.assert_equal(len(ifilenames), len(ofilenames))
    for x in xrange(len(ifilenames)):
        nibabel.load(ofilenames[x])


def test_niigz2nii_with_list_of_lists_of_filenames():
    # creates and save .nii.gz image
    ifilenames = []
    for i in xrange(4):
        img = create_random_image()
        ifilename = '/tmp/img%i.nii.gz' % i
        nibabel.save(img, ifilename)
        ifilenames.append(ifilename)

    # convert imgs to .nii
    ofilenames = niigz2nii([ifilenames], output_dir='/tmp/titi')

    # checks
    nose.tools.assert_equal(1, len(ofilenames))
    for x in xrange(len(ofilenames[0])):
        nibabel.load(ofilenames[0][x])


# run all tests
nose.runmodule(config=nose.config.Config(
        verbose=2,
        nocapture=True,
        ))
