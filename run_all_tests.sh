#!/bin/bash

echo "                                      ===pypreprocess (testing)==="
echo

nosetests -v $@ \  # always atleast level 1 verbose
coreutils/ \
algorithms/ \
spike/ \
external/nilearn/tests/test_datasets.py \
