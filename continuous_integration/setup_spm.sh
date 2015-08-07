#!/bin/bash
set -e

echo "0 is: $0"
# SPM_INSTALL_SCRIPT=$(dirname "$0")/install_spm.sh
SPM_INSTALL_SCRIPT=continuous_integration/install_spm.sh
echo "SPM_INSTALL_SCRIPT: $SPM_INSTALL_SCRIPT"
SPM_EXPORTS=$(sudo bash "$SPM_INSTALL_SCRIPT" | grep 'export SPM')
eval "$SPM_EXPORTS"

