#!/bin/bash

echo "                                      ===pypreprocess (testing)==="
echo

nosetests $@ \
coreutils/ \
algorithms/ \
spike/ \