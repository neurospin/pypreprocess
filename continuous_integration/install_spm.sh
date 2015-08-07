#!/bin/bash
set -e

SPM_ROOT_DIR=/opt/spm8
mkdir -p $SPM_ROOT_DIR && cd $SPM_ROOT_DIR
wget http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/spm8/spm8_r5236.zip
unzip -q spm8_r5236.zip
chmod 755 spm8/run_spm8.sh

wget http://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/MCR/glnxa64/MCRInstaller.bin
chmod 755 MCRInstaller.bin
./MCRInstaller.bin -P bean421.installLocation="mcr" -silent

cat <<EOF > $SPM_ROOT_DIR/spm8.sh
#!/bin/bash

SPM8_STANDALONE_HOME=$SPM_ROOT_DIR/spm8
exec "\${SPM8_STANDALONE_HOME}/run_spm8.sh" "\${SPM8_STANDALONE_HOME}/../mcr/v713" \${1+"\$@"}
EOF
chmod 755 $SPM_ROOT_DIR/spm8.sh

# Create CTF
$SPM_ROOT_DIR/spm8.sh quit

cat <<EOF
export SPM_DIR=$SPM_ROOT_DIR/spm8/spm8_mcr/spm8
export SPM_MCR=/opt/spm8/spm8.sh
EOF

