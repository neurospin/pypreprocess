#! /bin/bash
# Time-stamp: <2017-01-17 10:42:14 cp983411>
set -e

# set -x  # echo on

SPM_URL="http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/"
SPM12_FILE="spm12_r????.zip"
MCRINST=MCRInstaller.bin

SPM_ROOT_DIR=~/opt/spm12   #  local installation directory


mkdir -p $SPM_ROOT_DIR && cd $SPM_ROOT_DIR

if [ ! -d spm12 ]; then
    if [ ! -f "${SPM12_FILE}" ]; then
	wget --recursive --level=1 --no-directories --accept "${SPM12_FILE}" "${SPM_URL}"
	unzip -q "${SPM12_FILE}"
	chmod 755 spm12/run_spm12.sh
    else
	echo "An older version of spm12*.zip is already present in in current dir"
	echo "Please delete it before running this script"
	exit -1
    fi
fi

if [ ! -d mcr ]; then
    if [ ! -f MCRInstaller.bin ]; then
	wget "${SPM_URL}/MCR/glnxa64/${MCRINST}"
    fi
    chmod 755 ${MCRINST}
    ./${MCRINST} -P bean421.installLocation="mcr" -silent
fi


if [ ! -f $SPM_ROOT_DIR/spm12.sh ]; then
    cat <<EOF > $SPM_ROOT_DIR/spm12.sh
#!/bin/bash
SPM12_STANDALONE_HOME=$SPM_ROOT_DIR/spm12
exec "\${SPM12_STANDALONE_HOME}/run_spm12.sh" "\${SPM12_STANDALONE_HOME}/../mcr/v713" \${1+"\$@"}
EOF

    chmod 755 $SPM_ROOT_DIR/spm12.sh
fi

# Create CTF
$SPM_ROOT_DIR/spm12.sh quit
cmds="export SPM_DIR=$SPM_ROOT_DIR/spm12/; export SPM_MCR=$SPM_ROOT_DIR/spm12.sh"
${cmds}
echo "You may want to add the following commands (the exports) to your ~/.bashrc file once and for all."
echo
echo ${cmds}

