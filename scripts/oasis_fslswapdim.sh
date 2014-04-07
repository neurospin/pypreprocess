#!/usr/bash

# Out-of-the-box, he dimensions (i.e axis) of the images were permutated* xyz -> zyx, and
# the orientations (AP/PA, LR/RL, etc.) were also mixed up. Thusly, SPM was loading a
# shuffled version of the 3D array in memory, leading to exceptions in their linear algebra
# (singular operators, etc.).
#
# Fix is to run fslwapdim followed by fslorient -deleteorient.
#
# * This goes beyond the information coded by just the affine. This has to do with the
# [sq]form. 

DATA_DIR=$1
OUTPUT_DIR=$2

for subject_id in `ls ${DATA_DIR}`
do
    for in_file in `ls ${DATA_DIR}/${subject_id}/RAW/${suject_id}*anon.img`
    do
	mkdir -p ${OUTPUT_DIR}/${subject_id}
	out_file=${OUTPUT_DIR}/${subject_id}/`basename ${in_file%.*}_fslswapdim.nii.gz`
	cmd="fsl5.0-fslswapdim ${in_file} z x y ${out_file}"
	echo "Executing '${cmd}' ..."
	bash ${cmd}
	echo "Done."
    done
done
