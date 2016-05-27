#!/bin/bash
set -e

SPM_ROOT_DIR=~/opt/spm8

if [ -d "$SPM_ROOT_DIR" ]
then
    echo "spm already installed"
else
    echo "Creating directory : " $SPM_ROOT_DIR
    mkdir -p $SPM_ROOT_DIR && cd $SPM_ROOT_DIR
    echo "Downloading spm8 : "
    wget http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/spm8/spm8_r5236.zip
    echo "Unzipping : "
    unzip -q spm8_r5236.zip
    echo "Chmoding : "
    chmod 755 spm8/run_spm8.sh
    echo "Downloading MCR : "
    wget http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/MCR/glnxa64/MCRInstaller.bin
    echo "Chmoding : "
    chmod 755 MCRInstaller.bin
    echo "Installing MCR : "
    ./MCRInstaller.bin -P bean421.installLocation="mcr" -silent
fi

echo "Writing spm.sh : "
cat <<EOF > $SPM_ROOT_DIR/spm8.sh
#!/bin/bash
SPM8_STANDALONE_HOME=$SPM_ROOT_DIR/spm8
exec "\${SPM8_STANDALONE_HOME}/run_spm8.sh" "\${SPM8_STANDALONE_HOME}/../mcr/v713" \${1+"\$@"}
EOF
echo "Chmoding : "
chmod 755 $SPM_ROOT_DIR/spm8.sh
echo "Quiting : "
# Create CTF
$SPM_ROOT_DIR/spm8.sh quit
echo "Export SPM_DIR and SPM_MCR by running the following commands:"
echo
cmds="export SPM_DIR=$SPM_ROOT_DIR/spm8/spm8_mcr/spm8; export SPM_MCR=$SPM_ROOT_DIR/spm8.sh"
echo ${cmds}
echo
echo "N.B.: You may want add the above commands (the exports) to your ~/.bashrc file once and for all."
${cmds}










