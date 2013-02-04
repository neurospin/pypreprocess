import nibabel
import os
import joblib
import shutil
import commands


def is_3D(image_filename):
    return len(nibabel.load(image_filename).shape) == 3


def is_4D(image_filename):
    return len(nibabel.load(image_filename).shape) == 4


def delete_orientation(imgs, output_dir):
    """
    Function to delete (corrupt) orientation meta-data in nifti.

    XXX TODO: Do this without using fsl

    """

    output_imgs = []
    if not type(imgs) is list:
        imgs = [imgs]
    for img in imgs:
        output_img = os.path.join(
            output_dir, "deleteorient_" + os.path.basename(img))
        shutil.copy(img, output_img)
        print commands.getoutput(
            "fslorient -deleteorient %s" % output_img)
        print "+++++++Done (deleteorient)."
        print "Deleted orientation meta-data %s." % output_img
        output_imgs.append(output_img)

    return output_imgs


def do_3Dto4D_merge(threeD_img_filenames):
    """
    This function produces a single 4D nifti image from several 3D.

    """

    if type(threeD_img_filenames) is str:
        return threeD_img_filenames

    output_dir = os.path.dirname(threeD_img_filenames[0])

    # prepare for smart caching
    merge_cache_dir = os.path.join(output_dir, "merge")
    if not os.path.exists(merge_cache_dir):
        os.makedirs(merge_cache_dir)
    merge_mem = joblib.Memory(cachedir=merge_cache_dir, verbose=5)

    # merging proper
    fourD_img = merge_mem.cache(nibabel.concat_images)(threeD_img_filenames)
    fourD_img_filename = os.path.join(output_dir,
                                      "fourD_func.nii")

    # sanity
    if len(fourD_img.shape) == 5:
        fourD_img = nibabel.Nifti1Image(
            fourD_img.get_data()[..., ..., ..., 0, ...],
            fourD_img.get_affine())

    merge_mem.cache(nibabel.save)(fourD_img, fourD_img_filename)

    return fourD_img_filename
