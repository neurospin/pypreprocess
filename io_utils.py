"""
:Module: io_utils
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
