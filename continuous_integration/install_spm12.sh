#! /bin/bash
# Time-stamp: <2018-02-14 11:26:59 cp983411>

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

# Download
SPM_SRC=spm12_r????.zip
MCRINST=MCRInstaller.bin

wget -N -r -l1 --no-parent -nd  -P $SPM_ROOT_DIR -A.zip -R "index.html*" http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/

wget -N -r -l1 --no-parent -nd  -P $SPM_ROOT_DIR -A.bin http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/MCR/glnxa64/


# Install
cd $SPM_ROOT_DIR
unzip -q -u ${SPM_SRC}
chmod 755 spm12/run_spm12.sh

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

if [ ! -f /usr/lib/x86_64-linux-gnu/libXp.so.6 ]; then
    echo "WARNING!!!"
    echo "/usr/lib/x86_64-linux-gnu/libXp.so.6 is missing"
    echo 
    echo To install it:
    echo 'sudo add-apt-repository "deb http://securuty.ubuntu.com/ubuntu precise-security main"'
    echo 'sudo apt update'
    echo 'sudo apt install lixp6'
    echo 'sudo add-apt-repository -r "deb http://securuty.ubuntu.com/ubuntu precise-security main"'
fi

# Create CTF
${SPM_ROOT_DIR}/spm12.sh quit
cmds="export SPM_DIR=$SPM_ROOT_DIR/spm12/; export SPM_MCR=$SPM_ROOT_DIR/spm12.sh"
${cmds}
echo
echo ${cmds}
echo "IMPORTANT: you should now execute the following line: "
echo "echo ${cmds} >> $HOME/.bashrc"



