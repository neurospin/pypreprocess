#! /bin/bash
set -e

# set -x  # echo on

SPM_ROOT_DIR=~/opt/spm12   #  Installation directory

SPM_SRC=spm12_r7219.zip  # this used to be spm12_r6914.zip, watch out it might get pulled down again!!!
MCRINST=MCRInstaller.bin

mkdir -p $SPM_ROOT_DIR && cd $SPM_ROOT_DIR

if [ ! -d spm12 ]; then
    if [ ! -f ${SPM_SRC} ]; then
	wget http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/${SPM_SRC}
    fi
    unzip -q ${SPM_SRC}
    chmod 755 spm12/run_spm12.sh
fi

if [ ! -d mcr ]; then
    if [ ! -f MCRInstaller.bin ]; then
	wget http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/MCR/glnxa64/${MCRINST}
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
