"""
:Module: utils
:Synopsis: routine business related to image i/o manips
:Author: dohmatob elvis dopgima

"""

import os
import warnings
import re
import shutil
import tempfile
import filecmp
import numpy as np
import nibabel
import joblib
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.caching import Memory
from nilearn.image import iter_img
from nilearn.image.image import check_niimg, check_niimg_4d

DICOM_EXTENSIONS = [".dcm", ".ima", ".dicom"]


def is_niimg(img):
    """
    Checks whether given img is nibabel image object.

    """
    if isinstance(img, (nibabel.Nifti1Image,
                        nibabel.Nifti1Pair,
                        nibabel.Spm2AnalyzeImage,
                        nibabel.Spm99AnalyzeImage,
                        nibabel.MGHImage,
                        # add other supported image types below (e.g
                        # AnalyseImage, etc.)
                        )):
        return type(img)
    else:
        return False


def load_vols(niimgs):
    """Loads a nifti image (or a bail of) into a list qof 3D volumes.

    Parameters
    ----------
    niimgs: 3 or 4D Niimg-like object
        If niimgs is an iterable, checks if data is really 4D. Then,
        considering that it is a list of niimg and load them one by one.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if get_data
        and get_affine methods are present, raise an Exception otherwise.

    Returns
    -------
    niimgs_: list of nifti image objects
        The loaded volumes.
    """
    # try loading 4d
    try:
        niimgs = list(check_niimg_4d(niimgs, return_iterator=True))
    except TypeError:
        # probably not 4d
        niimgs = [check_niimg(niimgs)]
    except ValueError:
        # probably inconsisten affines
        pass
    try:
        # try loading volumes one-by-one
        if isinstance(niimgs, str): niimgs = [niimgs]
        return [check_niimg(niimg, ensure_ndim=3) for niimg in niimgs]
    except TypeError:
        pass

    # collect the loaded volumes into a list
    if is_niimg(niimgs):
        # should be 3d, squash 4th dimension otherwise
        if niimgs.shape[-1] == 1:
            return [nibabel.Nifti1Image(niimgs.get_data()[:, :, :, 0],
                                        niimgs.get_affine())]
        else:
            return list(iter_img(niimgs))
    else:
        niimgs = list(niimgs)
        if len(niimgs) == 1: niimgs = niimgs[0]
        return list(iter_img(niimgs))


def save_vols(vols, output_dir, basenames=None, affine=None,
              concat=False, prefix='', ext=None):
    """
    Saves a single 4D image or a couple of 3D vols unto disk.

    vols: single 4D nibabel image object, or list of 3D nibabel image objects
        volumes, of ndarray
        volumes to be saved

    output_dir: string
        existing filename, destination directory

    basenames: string or list of string, optional (default None)
        basename(s) for output image(s)

    affine: 2D array of shape (4, 4)
        affine matrix for the output images

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

    if not basenames is None:
        basenames = get_basenames(basenames, ext=ext)

    # sanitize output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # vols are ndarray ?
    if isinstance(vols, np.ndarray):
        vols = _nifti_or_ndarray_to_nifti(vols)

    # concat vols to single 4D film ?
    if concat:
        if isinstance(vols, list):
            vols = nibabel.concat_images([_nifti_or_ndarray_to_nifti(vol)
                                  for vol in vols],
                                         check_affines=False
                                         )
            if not basenames is None:
                if not isinstance(basenames, str):
                    basenames = basenames[0]
        else:
            if not basenames is None:
                if not isinstance(basenames, str):
                    raise RuntimeError(
                        ("concat=True specified but basenames is of type %s "
                         "instead of string") % type(basenames))

    if not isinstance(vols, list):
        if basenames is None:
            basenames = get_basenames("vols", ext=ext)

        if not isinstance(basenames, str):
            vols = nibabel.four_to_three(vols)
            filenames = []
            for vol, basename in zip(vols, basenames):
                if not isinstance(basename, str):
                    raise RuntimeError
                filename = os.path.join(output_dir, "%s%s" % (
                        prefix, basename))
                nibabel.save(vol, filename)
                filenames.append(filename)
        else:
            filenames = os.path.join(output_dir, "%s%s" % (
                    prefix, basenames))
            nibabel.save(vols, filenames)

        return filenames
    else:
        n_vols = len(vols)
        filenames = []
        if basenames is None:
            if prefix:
                prefix = prefix + "_"
        else:
            if isinstance(basenames, str):
                basenames = ["vol%i_%s" % (t, basenames)
                             for t in range(len(vols))]
            else:
                if len(set(basenames)) != len(vols):
                    raise RuntimeError
        for t, vol in zip(range(n_vols), vols):
            if isinstance(vol, np.ndarray):
                if affine is None:
                    raise ValueError(
                        ("vols is of type ndarray; you need to specifiy"
                         " `affine`"))
                else:
                    vol = nibabel.Nifti1Image(vol, affine)

            # save realigned vol unto disk
            if basenames is None:
                if ext is None:
                    ext = ".nii.gz"
                output_filename = os.path.join(output_dir,
                                               get_basename("%svol_%i" % (
                                                   prefix, t), ext=ext))
            else:
                basename = basenames if isinstance(
                    basenames, str) else basenames[t]
                output_filename = os.path.join(output_dir,
                                               get_basenames("%s%s" % (
                                                   prefix, basename), ext=ext))

            vol = check_niimg(vol)
            nibabel.save(vol, output_filename)

            # update rvols and filenames
            filenames.append(output_filename)

    return filenames


def save_vol(vol, output_filename=None, output_dir=None, basename=None,
             concat=False, **kwargs):
    """
    Saves a single volume to disk.

    """

    if not output_filename is None:
        nibabel.save(vol, output_filename)

        return output_filename
    else:
        if output_dir is None:
            raise ValueError(
                'One of output_filename and ouput_dir must be provided')

    if not basename is None:
        if isinstance(basename, list):
            basename = basename[:1]
        else:
            basename = [basename]

    # delegate to legacy save_vols
    return save_vols([vol], output_dir, basenames=basename,
                     concat=concat, **kwargs)[0]


def is_3D(image):
    """Check whether image is 3D"""

    if isinstance(image, str):
        if os.path.exists(image):
            image = nibabel.load(image)
        else:
            raise FileNotFoundError("File {} not found.".format(
                image))
    elif isinstance(image, list):
        image = nibabel.concat_images(image,
                                      check_affines=False
                                      )

    return len(image.shape) == 3


def is_4D(image):
    """Check whether image is 4D
    """

    if isinstance(image, str):
        image = nibabel.load(image)

    return len(image.shape) == 4


def get_vox_dims(niimg):
    """
    Infer voxel dimensions of a nifti image.

    Parameters
    ----------
    volume: string or nibabel image object, or list of
        input image whose voxel dimensions are to be computed

    Returns
    -------
    list of three floats

    """
    niimg = load_vols(niimg)[0]
    return [float(j) for j in niimg.get_header().get_zooms()[:3]]


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
    if isinstance(imgs, str):
        not_list = True
        imgs = [imgs]

    for img in imgs:
        output_img = os.path.join(output_dir,
                                  output_tag + os.path.basename(img))
        new_img = nibabel.load(img)
        new_img.set_sform(np.zeros((4, 4)))
        new_img.set_qform(np.eye(4))
        new_img.get_header()['dim_info'] = 0
        nibabel.save(new_img, output_img)
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

    if isinstance(threeD_img_filenames, str):
        return nibabel.load(threeD_img_filenames)

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # prepare for smart caching
    merge_cache_dir = os.path.join(output_dir, "merge")
    if not os.path.exists(merge_cache_dir):
        os.makedirs(merge_cache_dir)
    merge_mem = joblib.Memory(cachedir=merge_cache_dir, verbose=5)

    # merging proper
    fourD_img = merge_mem.cache(nibabel.concat_images)(threeD_img_filenames,
                                                       check_affines=False
                                                       )

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
                 new_vox_dims, output_filename=None):
    """
    Resamples an image to a new resolution

    Parameters
    ----------
    input_img_filename: string
        path to image to be resampled

    new_vox_dims: list or tuple of +ve floats
        new vox dimensions to which the image is to be resampled

    output_filename: string (optional)
        where output image will be written

    Returns
    -------
    output_filename: string
        where the resampled img has been written

    """

    try:
        from nilearn.image import resample_img as ni_resample_img
    except ImportError:
        raise RuntimeError(
            "nilearn not found on your system; can't do resampling!")

    # sanity
    if output_filename is None:
        output_filename = os.path.join(
            os.path.dirname(input_img_filename),
            "resample_" + os.path.basename(input_img_filename))

    # prepare for smart-caching
    output_dir = os.path.dirname(output_filename)
    cache_dir = os.path.join(output_dir, "resample_img_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mem = joblib.Memory(cachedir=cache_dir, verbose=5)

    # resample input img to new resolution
    resampled_img = mem.cache(ni_resample_img)(
        input_img_filename,
        target_affine=np.diag(new_vox_dims))

    # save resampled img
    nibabel.save(resampled_img, output_filename)

    return output_filename


def compute_mean_image(images, output_filename=None, threeD=False):
    """Computes the mean of --perhaps differently shaped-- images

    Parameters
    ----------
    images: string/image object, or list (-like) of
        image(s) whose mean we seek

    output_filename: string, optional (default None)
        output file where computed mean image will be written

    Returns
    -------
    mean nifti image object

    """

    # sanitize
    if not hasattr(images, '__iter__') or isinstance(images, str):
        images = [images]

    # make list of data an affines
    all_data = []
    all_affine = []
    for image in images:
        if not is_niimg(image):
            if isinstance(image, str):
                image = nibabel.load(image)
            else:
                image = nibabel.concat_images(image,
                                              check_affines=False
                                              )
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
    if isinstance(filenames, str):
        filenames = [filenames]
        if filenames[0].endswith(".img"):
            filenames.append(filenames[0].replace(".img", ".hdr"))
        if filenames[0].endswith(".hdr"):
            filenames.append(filenames[0].replace(".hdr", ".img"))

        hardlinked_filenames = [os.path.join(
            output_dir, os.path.basename(x)) for x in filenames]

        for src, dst in zip(filenames, hardlinked_filenames):
            if dst == src:
                # same file
                continue
            if not os.path.isfile(src):
                raise OSError("src file %s doesn't exist" % src)

            # do nothing if dst file already exists and is already a copy of
            # src
            if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=True):
                return dst

            # hard-link the file proper
            try:
                os.link(src, dst)
                print("\tHardlinked %s -> %s ..." % (src, dst))
            except OSError:
                # cross linking on different devices ?
                shutil.copy(src, dst)
                print("\tCopied %s -> %s" % (src, dst))

        return hardlinked_filenames[0]
    else:
        return [hard_link(_filenames, output_dir) for _filenames in filenames]


def get_basename(x, ext=None):
    bn = os.path.basename(x)
    if ext is None:
        return bn

    if not ext.startswith('.'):
        ext = "." + ext

    return "%s%s" % (bn.split(".")[0], ext)


def get_basenames(x, ext=None):
    if isinstance(x, list):
        return [get_basename(y, ext=ext) for y in x]
    elif isinstance(x, str):
        return get_basenames([x], ext=ext)[0]
    else:
        warnings.warn(
            "Input must be string or list of strings; got %s" % type(x))
        return None


def loaduint8(img, log=None):
    """Load data from file indicated by V into array of unsigned bytes.

    Parameters
    ----------
    img: string, `np.ndarray`, or niimg
        image to be loaded

    Returns
    -------
    uint8_data: `np.ndarray`, if input was ndarray; `nibabel.NiftiImage1' else
        the loaded image (dtype='uint8')

    """

    def _progress_bar(msg):
        """
        Progress bar.

        """

        if log is not None:
            log(msg)
        else:
            print(msg)

    _progress_bar("Loading %s..." % img)

    # load volume into memory
    img = load_vols(img)[0]
    vol = img.get_data()

    # if isinstance(img, np.ndarray) or isinstance(img, list):
    #     vol = np.array(img)
    # elif isinstance(img, str):
    #     img = nibabel.load(img)
    #     vol = img.get_data()
    # elif is_niimg(img):
    #     vol = img.get_data()
    # else:
    #     raise TypeError("Unsupported input type: %s" % type(img))

    # if vol.ndim == 4:
    #     vol = vol[..., 0]

    # assert vol.ndim == 3

    def _get_slice(p):
        """
        Gets data pth slice of vol.

        """

        return vol[..., p].copy()

    # min/max
    mx = -np.inf
    mn = np.inf
    _progress_bar("\tComputing min/max...")
    for p in range(vol.shape[2]):
        _img = _get_slice(p)
        mx = max(_img.max(), mx)
        mn = min(_img.min(), mn)

    # load data from file indicated by V into an array of unsigned bytes
    uint8_dat = np.ndarray(vol.shape, dtype='uint8')
    for p in range(vol.shape[2]):
        _img = _get_slice(p)

        # pth slice
        uint8_dat[..., p] = np.uint8(np.maximum(np.minimum(np.round((
                            _img - mn) * (255. / (mx - mn))), 255.), 0.))

    _progress_bar("...done.")

    # return the data
    if isinstance(img, str) or is_niimg(img):
        return nibabel.Nifti1Image(uint8_dat, img.get_affine())
    else:
        return uint8_dat


def ravel_filenames(fs):
    if isinstance(fs, str):
        ofilenames = fs
        file_types = 'str'
    else:
        file_types = []
        ofilenames = []
        for x in fs:
            if isinstance(x, str):
                ofilenames.append(x)
                file_types.append('str')
            else:
                ofilenames += x
                file_types.append(('list', len(x)))

    return ofilenames, file_types


def unravel_filenames(filenames, file_types):
    if not isinstance(file_types, str):
        _tmp = []
        s = 0
        for x in file_types:
            if x == 'str':
                if isinstance(filenames, str):
                    _tmp = filenames
                    break
                else:
                    _tmp.append(
                        filenames[s])
                    s += 1
            else:
                _tmp.append(
                    filenames[s: s + x[1]])
                s += x[1]

        filenames = _tmp

    return filenames


def niigz2nii(ifilename, output_dir=None):
    """
    Converts .nii.gz to .nii (SPM doesn't support .nii.gz images).

    Parameters
    ----------
    ifilename: string, of list of strings
        input filename of image to be extracted

    output_dir: string, optional (default None)
        output directory to which exctracted file will be written.
        If no value is given, the output will be written in the parent
        directory of ifilename

    Returns
    -------
    ofilename: string
        filename of extracted image

    """

    if isinstance(ifilename, list):
        return [niigz2nii(x, output_dir=output_dir) for x in ifilename]
    else:
        if not isinstance(ifilename, str):
            raise RuntimeError(
                "ifilename must be string or list of strings, got %s" % type(
                    ifilename))

    if not ifilename.endswith('.nii.gz'):
        return ifilename

    ofilename = ifilename[:-3]
    if not output_dir is None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ofilename = os.path.join(output_dir, os.path.basename(ofilename))

    nibabel.save(nibabel.load(ifilename), ofilename)

    return ofilename


def isdicom(source_name):
    """
    Determines, by filename extension, whether `source_name` is a DICOM file
    or not.

    Returns
    -------
    True if `source_name` is a DICOM file, False otherwise.

    """

    if is_niimg(source_name):
        return False

    for ext in DICOM_EXTENSIONS:
        if source_name.lower().endswith(ext):
            return True

    return False


def dcm2nii(source_names, terminal_output="allatonce", gzip_output=False,
            anonymize=True, output_dir=None, caching=True,
            **other_dcm2nii_kwargs):
    """
    Converts DICOM (dcm) images to Nifti.

    """

    # all input files must be DICOM
    if is_niimg(source_names):
        return source_names, None

    for source_name in [source_names] if isinstance(
        source_names, str) else source_names:
        if not isdicom(source_name):
            return source_names, None  # not (all) DICOM; nothx to do

    # sanitize output dir
    if output_dir is None:
        output_dir = os.path.dirname(source_names if isinstance(
                source_names, str) else source_names[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare node
    if caching:
        cache_dir = os.path.join(output_dir, "cache_dir")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        dcm2nii_node = Memory(cache_dir).cache(Dcm2nii)
    else:
        dcm2nii_node = Dcm2nii.run

    # run node and collect results
    dcm2nii_result = dcm2nii_node(source_names=source_names,
                                  terminal_output=terminal_output,
                                  anonymize=anonymize,
                                  gzip_output=gzip_output,
                                  output_dir=output_dir,
                                  **other_dcm2nii_kwargs)

    return dcm2nii_result.outputs.converted_files, dcm2nii_result


def _expand_path(path, relative_to=None):
    """
    Expands a path, doing replacements like ~ -> $HOME, . -> cwd, etc.

    Returns
    -------
    _path: string
        expanded path, or None if path doesn't exist

    """

    # cd to reference directory
    if relative_to is None:
        relative_to = os.getcwd()
    else:
        relative_to = _expand_path(relative_to)
        if not os.path.exists(relative_to):
            raise OSError(
                "Reference path %s doesn't exist" % relative_to)
    old_cwd = os.getcwd()
    os.chdir(relative_to)

    _path = path
    if _path.startswith(".."):
        if _path == "..":
            _path = os.path.dirname(os.getcwd())
        else:
            match = re.match("(?P<head>(?:\.{2}\/)+)(?P<tail>.*)", _path)
            if match:
                _path = os.getcwd()
                for _ in range(len(match.group("head")) // 3):
                    _path = os.path.dirname(_path)
                _path = os.path.join(_path, match.group("tail"))
            else:
                _path = None
    elif _path.startswith("./"):
        _path = _path[2:]
    elif _path.startswith("."):
        _path = _path[1:]
    elif _path.startswith("~"):
        if _path == "~":
            _path = os.environ["HOME"]
        else:
            _path = os.path.join(os.environ["HOME"], _path[2:])

    if not _path is None:
        _path = os.path.abspath(_path)

    # restore cwd
    os.chdir(old_cwd)

    return _path


def get_relative_path(ancestor, descendant):
    """
    Get's path of a file or directory (descendant) relative to another
    directory (ancestor). For example get_relative_path("/toto/titi",
    "/toto/titi/tata/test.txt") should return "tata/test.txt"

    """

    if ancestor == descendant:
        return ""

    ancestor = ancestor.rstrip("/")
    descendant = descendant.rstrip("/")
    match = re.match(r'%s\/(.*)' % ancestor, descendant)
    if match is None:
        return None
    else:
        return match.group(1)


def get_shape(img):
    """
    Computes shape of an image.

    Parameters
    ----------
    img: niim, string, or list of such
        img whose shape is sought-for

    Returns
    -------
    shape: tuple
        shape of image


    """

    if isinstance(img, str):
        return nibabel.load(img).shape
    elif is_niimg(img):
        return img.shape
    else:
        return tuple(list(get_shape(img[0])) + [len(img)])


def compute_output_voxel_size(img, voxel_size):
    """
    Computes desired output voxel size of an img.

    """

    if voxel_size in ['original', 'auto']:
        # write original voxel size
        return get_vox_dims(img)
    elif not voxel_size is None:
        # set output voxel size to specified value
        return voxel_size
    else:
        # donno
        return None


def sanitize_fwhm(fwhm):
    if fwhm is None: return 0.  # no smoothing
    if not np.shape(fwhm):
        fwhm = [fwhm, fwhm, fwhm]
    if len(fwhm) == 1:
        fwhm = list(fwhm) * 3
    else:
        if len(fwhm) != 3:
            raise ValueError("fwhm must be float or list of 3 "
                             "floats; got %s" % fwhm)
    return fwhm


def nii2niigz(input_filename, output_dir=None):
    """
    Converts .nii to .nii.gz

    Parameters
    ----------

    input_filename: string, of list of strings
        input filename of image to be compressed

    output_dir: string, optional (default None)
        output directory to which compressed file will be written.
        If no value is given, the output will be written in the parent
        directory of input_filename

    Returns
    -------
    output_filename: string
        filename of extracted image

    """

    if isinstance(input_filename, list):
        return [nii2niigz(x, output_dir=output_dir) for x in input_filename]
    else:
        if not isinstance(input_filename, str):
            raise RuntimeError(
                "input_filename must be string"
                "or list of strings, got %s" % type(
                    input_filename))

    if not input_filename.endswith('.nii'):
        return input_filename

    output_filename = input_filename + '.gz'
    if output_dir is None:
        output_dir = os.path.dirname(input_filename)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    output_filename = os.path.join(output_dir,
                                   os.path.basename(output_filename))

    nibabel.save(nibabel.load(input_filename), output_filename)

    return output_filename
