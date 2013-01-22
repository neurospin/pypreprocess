import nibabel
import os
import joblib


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
