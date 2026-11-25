#!/bin/zsh
# args: k L nsamples chunkid
PY="${PYTHON:-python}"
$PY sim.py $1 $2 $3 $((1000*$1+10*$2+$4)) data/k$1_L$2_c$4.npy
