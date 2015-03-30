import os
import glob
import re
import nibabel as nb


def unzip_nii_gz(dirname, output_dir=None):
    """
    Helper function for extracting .nii.gz to .nii.

    """

    for filename in glob.glob('%s/*.nii.gz' % dirname):
        if not os.path.exists(re.sub("\.gz", "", filename)):
            img = nb.load(filename)
            folder, fname = os.path.split(filename)
            fname = fname.split('.nii.gz')[0]
            if output_dir is None:
                nb.save(img, os.path.join(folder, '%s.nii' % fname))
            else:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            nb.save(img, os.path.join(output_dir, '%s.nii' % fname))
