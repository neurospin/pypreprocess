#! /bin/bash
# Time-stamp: <2017-06-04 16:02:14 cp983411>

# Download and install SPM12 standalone (no Matlab license required)
# see https://en.wikibooks.org/wiki/SPM/Standalone

set -e
# set -x  # echo on for debugging

#  Installation directory can be specified as first argument on the command line
#  Warning: use a fully qualitifed path (from root) to correctly set up env variables

if [ $# -eq 0 ]
  then
      SPM_ROOT_DIR=$HOME/opt/spm12   # default
else
      SPM_ROOT_DIR=$1
fi

mkdir -p $SPM_ROOT_DIR

SPM_ROOT_DIR=~/opt/spm12   #  Installation directory

SPM_SRC=spm12_r7219.zip  # this used to be spm12_r6914.zip, watch out it might get pulled down again!!!
MCRINST=MCRInstaller.bin

wget -N -r -l1 --no-parent -nd  -P $SPM_ROOT_DIR -A.zip -R "index.html*" http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/

wget -N -r -l1 --no-parent -nd  -P $SPM_ROOT_DIR -A.bin http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/MCR/glnxa64/


if [ ! -d spm12 ]; then
    if [ ! -f ${SPM_SRC} ]; then
	wget http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/${SPM_SRC}
    fi
fi

if [ ! -d mcr ]; then
   chmod 755 ${MCRINST}
   ./${MCRINST} -P bean421.installLocation="mcr" -silent
fi
   
# create start-up script
cat <<EOF > spm12.sh
#!/bin/bash
SPM12_STANDALONE_HOME=$SPM_ROOT_DIR/spm12
exec "\${SPM12_STANDALONE_HOME}/run_spm12.sh" "\${SPM12_STANDALONE_HOME}/../mcr/v713" \${1+"\$@"}
EOF

chmod 755 spm12.sh

# Create CTF
${SPM_ROOT_DIR}/spm12.sh quit
cmds="export SPM_DIR=$SPM_ROOT_DIR/spm12/; export SPM_MCR=$SPM_ROOT_DIR/spm12.sh"
${cmds}
echo "You may want to add the following commands (the exports) to your ~/.bashrc file once and for all."
echo
echo ${cmds}
