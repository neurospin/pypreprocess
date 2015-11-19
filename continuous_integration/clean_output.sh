#!/bin/bash
set -e

# A script to delete cache, temporary and heavy files
# for a given pypreprocess output


remove_files(){
  exts="*.nii *.nii.gz *.img *.hdr *.txt"
  for ext in $exts
  do
    echo $ext $1
    find $1 -name $ext -exec rm -rf {} \;
  done 
}

remove_dirs(){
  dirs="variance_maps effects_maps t_maps z_maps tmp Session1 Session2 QA cache_dir"
  for dir in $dirs
  do
    p="$(find $1 -name $dir -type d)"
    if [ ! -z "$p" ]
    then
      rm -rf $p
      echo $p" deleted"
    fi
  done
  find $1 -type d -empty -delete
}

# Main
## Directories to clean
paths="/home/ubuntu/nilearn_data/spm_auditory/pypreprocess_output/"
paths="$paths /home/ubuntu/nilearn_data/spm_multimodal_fmri/pypreprocess_output/"
paths="$paths /home/ubuntu/nilearn_data/fsl_feeds/pypreprocess_output/"

## Main loop
for path in $paths
do
  if [ -d "$path" ]
  then
    echo "Cleaning "$path
    remove_files $path
    remove_dirs $path
  else
    echo $path" not found"
  fi
done

