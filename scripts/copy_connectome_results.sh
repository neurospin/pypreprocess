#!/bin/bash

SRC_DIR=/volatile/home/edohmato/connectome_output/hcp_preproc/MOTOR
DST_DIR=/media/DISK1S1/connectome-beta-maps-HCP-mni

for subject_id in $(ls ${SRC_DIR})
do
    # copy mask
    if [ -f ${SRC_DIR}/${subject_id}/mask.nii.gz ]; then
	if [ ! -f ${DST_DIR}/${subject_id}/mask.nii.gz ]; then
	    echo
	    echo "Copying ${SRC_DIR}/${subject_id}/mask.nii.gz -> ${DST_DIR}/${subject_id} ..."
	    mkdir -p ${DST_DIR}/${subject_id}
	    cp -a ${SRC_DIR}/${subject_id}/mask.nii.gz ${DST_DIR}/${subject_id}
	    echo "... done"
	    echo
	fi
    fi

    # copy maps
    if [ -d ${SRC_DIR}/${subject_id}/with_preproc_from_HCP/effects_maps ]; then
	if [ ! -d ${DST_DIR}/${subject_id}/effects_maps ]; then
	    echo
	    echo "Copying ${SRC_DIR}/${subject_id}/with_preproc_from_HCP/effects_maps -> ${DST_DIR}/${subject_id} ..."
	    mkdir -p ${DST_DIR}/${subject_id}
	    cp -a ${SRC_DIR}/${subject_id}/with_preproc_from_HCP/effects_maps ${DST_DIR}/${subject_id}
	    echo "... done"
	    echo
	fi
    fi
done
