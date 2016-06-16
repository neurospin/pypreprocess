#! /usr/bin/env python
# Time-stamp: <2016-05-31 14:58:27 chrplr>
#

""" Extracts slice timing information from a dicom file.
To be passed  as argument to slice_order for slice timing correction algorithms """

from __future__ import print_function
import os
import glob
import dicom
import sys

try:
    dicom_path = sys.argv[1]

    if os.path.isdir(dicom_path):
        dicom_ref = sorted(glob.glob(os.path.join(dicom_path, '*.dcm')))[4]
    else:
        if (os.path.isfile(dicom_path)):
            dicom_ref = dicom_path

    TR = dicom.read_file(dicom_ref).RepetitionTime
    slice_times = dicom.read_file(dicom_ref)[0x19, 0x1029].value
    nb_slices = len(slice_times)

except:
    print("Unexpected error: %s" % sys.exc_info()[0])
    print("\nUsage:\n    %s dicom_path\nWhere dicom_path is a dicom directory" % sys.argv[0])
    sys.exit(-1)

print("TR = %.3f" % (TR/1000.))
#print("nb_slices = %d" % nb_slices)
print("slice_timings = ", end=" ")
for v in slice_times:
    print("%.1f" % v, end=" ")
print("\n", end="")
print("refslice = %.3f" % (TR/2000.))

