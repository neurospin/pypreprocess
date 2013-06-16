import os
import gzip
import glob
import re


def unzip_nii_gz(dirname):
    """
    Helper function for extracting .nii.gz to .nii.

    """

    for filename in glob.glob('%s/*.nii.gz' % dirname):
        if not os.path.exists(re.sub("\.gz", "", filename)):
            f_in = gzip.open(filename, 'rb')
            f_out = open(filename[:-3], 'wb')
            f_out.writelines(f_in)
            f_out.close()
            f_in.close()
            # os.remove(filename)  # XXX why ?
