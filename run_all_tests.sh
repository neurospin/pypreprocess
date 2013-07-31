#!/bin/bash

echo "                                      ===pypreprocess (testing)==="
echo

nosetests -v $@ coreutils/ \
spike/ \
external/nilearn/tests/test_datasets.py \
algorithms/ \
