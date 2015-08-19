#!/bin/bash
set -e

SPM_INSTALL_SCRIPT=continuous_integration/install_spm.sh
echo ""
echo "SPM_INSTALL_SCRIPT: $SPM_INSTALL_SCRIPT"
sudo bash $SPM_INSTALL_SCRIPT
