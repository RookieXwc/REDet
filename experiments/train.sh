#!/bin/bash

ROOT=../../
T=`date +%m%d%H%M`
export ROOT=$ROOT
cfg=configs/Table1_r50_REDet.yaml
export PYTHONPATH=$ROOT:$PYTHONPATH
python -m up train \
  --nm=1 \
  --ng=8 \
  --launch=pytorch \
  --config=$cfg \
  2>&1 | tee experiments/train_log/log.train.$T.$(basename $cfg) 
