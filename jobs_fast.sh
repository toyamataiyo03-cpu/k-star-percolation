#!/bin/zsh
# args: k L nsamples chunk
PY="${PYTHON:-python}"
$PY fast.py sim $1 $2 $3 $((70000 + 100*$1 + 7*$2 + $4)) data2/k$1_L$2_c$4.npy
