#!/bin/bash

echo "                                      ===pypreprocess (testing)==="
echo

nosetests -v $@ coreutils/ \
external/nilearn/tests/test_datasets.py \
algorithms/ \
