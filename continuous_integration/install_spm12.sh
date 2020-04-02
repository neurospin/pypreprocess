#! /bin/bash
# Time-stamp: <2020-03-13 11:26:59 cp983411>

# Download and install SPM12 standalone (no Matlab license required)

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
mkdir -p $SPM_ROOT_DIR/mcr

# Download
SPM_SRC=spm12_r7771_Linux_R2019b.zip
MCRINST=MATLAB*.zip

wget -N -r -l1 --no-parent -nd  -P $SPM_ROOT_DIR https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/dev/$SPM_SRC --no-check-certificate
wget -N -r -l1 --no-parent -nd  -P $SPM_ROOT_DIR/mcr https://ssd.mathworks.com/supportfiles/downloads/R2019b/Release/4/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2019b_Update_4_glnxa64.zip --no-check-certificate

# Install SPM
cd $SPM_ROOT_DIR
unzip -q -u ${SPM_SRC}
chmod 755 spm12/run_spm12.sh

# Install Matlab runtime compiler
cd $SPM_ROOT_DIR/mcr
unzip ${MCRINST}
./install -mode silent -agreeToLicense yes -destinationFolder $SPM_ROOT_DIR/mcr -outputFile $SPM_ROOT_DIR/mcr

# create start-up script
cd $SPM_ROOT_DIR
cat <<EOF > spm12.sh
#!/bin/bash
SPM12_STANDALONE_HOME=$SPM_ROOT_DIR/spm12
exec "\${SPM12_STANDALONE_HOME}/run_spm12.sh" "\${SPM12_STANDALONE_HOME}/../mcr/v97" \${1+"\$@"}
EOF

chmod 755 spm12.sh

if [ ! -f /usr/lib/x86_64-linux-gnu/libXp.so.6 ]; then
    echo "WARNING!!!"
    echo "/usr/lib/x86_64-linux-gnu/libXp.so.6 is missing"
    echo
    echo To install it:
    echo 'sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu precise-security main"'
    echo 'sudo apt update'
    echo 'sudo apt install lixp6'
    echo 'sudo add-apt-repository -r "deb http://security.ubuntu.com/ubuntu precise-security main"'
fi

# Create CTF
${SPM_ROOT_DIR}/spm12.sh quit
cmds="export SPM_DIR=$SPM_ROOT_DIR/spm12/; export SPM_MCR=$SPM_ROOT_DIR/spm12.sh"
${cmds}
echo
echo ${cmds}
echo "IMPORTANT: you should now execute the following line: "
echo "echo ${cmds} >> $HOME/.bashrc"
