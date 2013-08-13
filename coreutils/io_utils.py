"""
:Module: utils
:Synopsis: routine business related to image i/o manips
:Author: dohmatob elvis dopgima

"""

import os
import joblib
import commands
import tempfile
import numpy as np
import nibabel
from external.nilearn import resampling


def is_niimg(img):
    """
    Checks whether given img is nibabel image object.

    """

    if isinstance(img, (nibabel.Nifti1Image,
                        nibabel.Nifti1Pair
                        # add other supported image types below (e.g
                        # AnalyseImage, etc.)
                        )):
        return type(img)
    else:
        return False


def _load_vol(x):
    """
    Loads a single 3D volume.

    Parameters
    ----------
    x: string (existent filename) or nibabel image object
        image to be loaded

    Returns
    -------
    vol: nibabel image object

    """

    if isinstance(x, basestring):
        vol = nibabel.load(x)
    elif is_niimg(x):
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


def _load_specific_vol(vols, t):
    """Utility function for loading specific volume on demand.

    Parameters
    ----------
    vols: string(s) or nibabel image object(s)

    """

    assert t >= 0

    if isinstance(vols, list):
        n_scans = len(vols)
        vol = _load_vol(vols[t])
    elif is_niimg(vols) or isinstance(vols, basestring):
        _vols = nibabel.load(vols) if isinstance(vols, basestring) else vols
        if len(_vols.shape) != 4:
            raise ValueError(
                "Expecting 4D image, got %iD" % len(_vols.shape))

        n_scans = _vols.shape[-1]
        vol = nibabel.four_to_three(_vols)[t]
    else:  # unhandled type
        raise TypeError(
            ("vols must be string, image object, or list of such; "
             "got %s" % type(vols)))

    if len(vol.shape) == 4:
        vol = nibabel.Nifti1Image(vol.get_data()[..., ..., ..., 0],
                                  vol.get_afffine())

    return vol, n_scans


def three_to_four(images):
    if is_niimg(images):
        return images

    if isinstance(images, basestring):
        return nibabel.load(images)

    vol_0 = _load_vol(images[0])

    data = np.ndarray(list(vol_0.shape) + [len(images)],
                      dtype=vol_0.get_data().dtype)
    data[..., 0] = vol_0.get_data()

    for t in xrange(1, len(images)):
        vol_t = _load_vol(images[t])
        assert vol_t.shape == vol_0.shape

        data[..., t] = vol_t.get_data()

    if data.ndim == 5:
        assert data.shape[-1] == 1
        data = data[..., 0]

    return nibabel.Nifti1Image(data, vol_0.get_affine())


def _save_vols(vols, output_dir, basenames=None, affine=None,
               concat=False, prefix='', ext='.nii.gz'):
    """
    Saves a single 4D image or a couple of 3D vols unto disk.

    vols: single 4D nibabel image object, or list of 3D nibabel image objects
        volumes to be saved
    output_dir: string
        existing filename, destination directory
    basenames: string or list of string, optional (default None)
        basename(s) for output image(s)
    concat: bool, optional (default False)
        concatenate all vols into a single film
    prefix: string, optional (default '')
       prefix to be prepended to output file basenames
    ext: string, optional (default ".nii.gz")
        file extension for output images

    Returns
    -------
    string of list of strings, dependending on whether vols is list or
    not, and on whether concat is set or not
        the output image filename(s)

    """

    def _nifti_or_ndarray_to_nifti(x):
        if is_niimg(x):
            if not affine is None:
                raise ValueError(
                    ("vol is of type %s; not expecting `affine` parameter."
                     ) % type(x))
            else:
                return x

        if affine is None:
            raise ValueError(
                "vol is of type ndarray; you need to specifiy `affine`")
        else:
            return nibabel.Nifti1Image(x, affine)

    if isinstance(vols, np.ndarray):
        vols = _nifti_or_ndarray_to_nifti(vols)

    if concat:
        if isinstance(vols, list):
            vols = nibabel.concat_images([_nifti_or_ndarray_to_nifti(vol)
                                          for vol in vols])
            if not basenames is None:
                basenames = basenames[0]

    if not isinstance(vols, list):
        if basenames is None:
            basenames = "vols"

        filenames = os.path.join(output_dir, "%s%s%s" % (
                prefix, basenames.split(".")[0], ext))
        nibabel.save(vols, filenames)

        return filenames
    else:
        n_vols = len(vols)
        filenames = []

        if basenames is None:
            if prefix:
                prefix = prefix + "_"
        else:
            assert not isinstance(basenames, basestring)
            assert len(basenames) == len(vols)

        for t, vol in zip(xrange(n_vols), vols):
            if isinstance(vol, np.ndarray):
                if affine is None:
                    raise ValueError(
                        ("vols is of type ndarray; you need to specifiy"
                         " `affine`"))
                else:
                    vol = nibabel.Nifti1Image(vol, affine)

            # save realigned vol unto disk
            if basenames is None:
                output_filename = os.path.join(output_dir,
                                               "%svol_%i%s" % (
                        prefix, t, ext))
            else:
                output_filename = os.path.join(output_dir, "%s%s%s" % (
                        prefix, basenames[t].split(".")[0], ext))

            nibabel.save(vol, output_filename)

            # update rvols and filenames
            filenames.append(output_filename)

    return filenames


def _save_vol(vol, output_dir, basenames=None, concat=False, **kwargs):
    if not basenames is None:
        basenames = [basenames]

    return _save_vols([vol], output_dir, basenames=basenames,
                      concat=False, **kwargs)[0]


def is_3D(image):
    """Check whether image is 3D"""

    if isinstance(image, basestring):
        image = nibabel.load(image)
    elif isinstance(image, list):
        image = nibabel.concat_images(image)

    if len(image.shape) == 3:
        return True
    else:
        return len(image.shape) == 4 and image.shape[-1] == 1


def is_4D(image):
    """Check whether image is 4D
    """

    if isinstance(image, basestring):
        image = nibabel.load(image)

    if len(image.shape) == 4:
        return True
    else:
        return len(image.shape) == 5 and image.shape[-1] == 1


def get_vox_dims(volume):
    """
    Infer voxel dimensions of a nifti image.

    Parameters
    ----------
    volume: string
        image whose voxel dimensions we seek

    """

    if not isinstance(volume, basestring):
        volume = volume[0]
    try:
        nii = nibabel.load(volume)
    except:
        # XXX quick and dirty
        nii = nibabel.concat_images(volume)

    hdr = nii.get_header()
    voxdims = hdr.get_zooms()

    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]


def delete_orientation(imgs, output_dir, output_tag=''):
    """Function to delete (corrupt) orientation meta-data in nifti

    XXX TODO: Do this without using fsl

    Parameters
    ----------
    imgs: string or list of string
       path (paths) to 4D (3D) image (images) under inspection

    output_dir: string
       directory to which output will be written

    output_tag: string (optional)
       tag to append output image filename(s) with

    Returns
    -------
    output images

    """

    output_imgs = []
    not_list = False
    if not type(imgs) is list:
        not_list = True
        imgs = [imgs]

    for img in imgs:
        output_img = os.path.join(
            output_dir,
            "deleteorient_%s_" % (output_tag) + os.path.basename(img))
        nibabel.save(nibabel.load(img), output_img)
        print commands.getoutput(
            "fslorient -deleteorient %s" % output_img)
        print "+++++++Done (deleteorient)."
        print "Deleted orientation meta-data %s." % output_img
        output_imgs.append(output_img)

    if not_list:
        output_imgs = output_imgs[0]

    return output_imgs


def do_3Dto4D_merge(
    threeD_img_filenames,
    output_dir=None,
    output_filename=None):
    """
    This function produces a single 4D nifti image from several 3D.

    threeD_img_filenames: list of string
        paths to images to be merged

    Returns
    -------
    returns nifit image object

    """

    if isinstance(threeD_img_filenames, basestring):
        return nibabel.load(threeD_img_filenames)

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # prepare for smart caching
    merge_cache_dir = os.path.join(output_dir, "merge")
    if not os.path.exists(merge_cache_dir):
        os.makedirs(merge_cache_dir)
    merge_mem = joblib.Memory(cachedir=merge_cache_dir, verbose=5)

    # merging proper
    fourD_img = merge_mem.cache(nibabel.concat_images)(threeD_img_filenames)

    # sanity
    if len(fourD_img.shape) == 5:
        fourD_img = nibabel.Nifti1Image(
            fourD_img.get_data()[..., ..., ..., 0, ...],
            fourD_img.get_affine())

    # save image to disk
    if not output_filename is None:
        merge_mem.cache(nibabel.save)(fourD_img, output_filename)

    return fourD_img


def resample_img(input_img_filename,
                 new_vox_dims, output_img_filename=None):
    """Resamples an image to a new resolution

    Parameters
    ----------
    input_img_filename: string
        path to image to be resampled

    new_vox_dims: list or tuple of +ve floats
        new vox dimensions to which the image is to be resampled

    output_img_filename: string (optional)
        where output image will be written

    Returns
    -------
    output_img_filename: string
        where the resampled img has been written

    """

    # sanity
    if output_img_filename is None:
        output_img_filename = os.path.join(
            os.path.dirname(input_img_filename),
            "resample_" + os.path.basename(input_img_filename))

    # prepare for smart-caching
    output_dir = os.path.dirname(output_img_filename)
    cache_dir = os.path.join(output_dir, "resample_img_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = joblib.Memory(cachedir=cache_dir, verbose=5)

    # resample input img to new resolution
    resampled_img = mem.cache(resampling.resample_img)(
        input_img_filename,
        target_affine=np.diag(new_vox_dims))

    # save resampled img
    nibabel.save(resampled_img, output_img_filename)

    return output_img_filename


def compute_mean_image(images, output_filename=None, threeD=False):
    """Computes the mean of --perhaps differently shaped-- images

    Parameters
    ----------
    images: string/image object, or list (-like) of
        image(s) whose mean we seek

    Returns
    -------
    mean nifti image object

    """

    # sanitize
    if not hasattr(images, '__iter__') or isinstance(images, basestring):
        images = [images]

    # make list of data an affines
    all_data = []
    all_affine = []
    for image in images:
        if isinstance(image, basestring):
            image = nibabel.load(image)
        else:
            image = nibabel.concat_images(image)
            # raise IOError(type(image))
        data = image.get_data()

        if threeD:
            if is_4D(image):
                data = data.mean(-1)

        all_data.append(data)
        all_affine.append(image.get_affine())

    # compute mean
    mean_data = np.mean(all_data, axis=0)

    # XXX I'm assuming all the affines are equal
    mean_affine = all_affine[0]

    mean_image = nibabel.Nifti1Image(mean_data, mean_affine)

    # save mean image
    if output_filename:
        nibabel.save(mean_image, output_filename)

    # return return result
    return mean_image


def compute_mean_3D_image(images, output_filename=None):
    """Computes the mean of --perhaps differently shaped-- images

    Parameters
    ----------
    images: string/image object, or list (-like) of
        image(s) whose mean we seek

    Returns
    -------
    mean nifti image object

    """

    return compute_mean_image(images, output_filename=output_filename,
                              threeD=True)


def hard_link(filenames, output_dir):
    """
    Auxiliary function for hardlinking files to specified output director.

    Parameters
    ----------
    filenames: string, list of strings, or list of such, or list of such,
    or list of such, and so on.
        files to hard-link
    output_dir: string
        output directory to which the files will be hard-linked

    Returns
    -------
    hardlinked_filenames: same structure as the input filenames
        the hard-linked filenames

    """

    if isinstance(filenames, basestring):
        filenames = [filenames]
        if filenames[0].endswith(".img"):
            filenames.append(filenames[0].replace(".img", ".hdr"))
        if filenames[0].endswith(".hdr"):
            filenames.append(filenames[0].replace(".hdr", ".img"))

        hardlinked_filenames = [os.path.join(
            output_dir, os.path.basename(x)) for x in filenames]

        for x, y in zip(filenames, hardlinked_filenames):
            print "\tHardlinking %s -> %s..." % (x, y)

            # unlink if link already exists
            if os.path.exists(y):
                os.unlink(y)

            # hard-link the file proper
            os.link(x, y)

        return hardlinked_filenames[0]
    else:
        return [hard_link(_filenames, output_dir) for _filenames in filenames]


def get_basenames(x):
    if isinstance(x, list):
        return [os.path.basename(y) for y in x]
    elif isinstance(x, basestring):
        return os.path.basename(x)
    else:
        raise TypeError(
            "Input must be string or list of strings; got %s" % type(x))
