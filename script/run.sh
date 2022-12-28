#!/usr/bin/env bash
set -e
cd ..
export PYTHONPATH=`pwd`:$PYTHONPATH

WORK_DIR=$(pwd)
SRC_DIR="${WORK_DIR}/src"

python "${SRC_DIR}"/main.py train\
  --desc='Description of experiment' \
  --cuda=0 \
  --dataset='Gallbladder'\
  --model='se_resnet' \
  --action='base' \
  --epoch=100 \
  --batch_size=64 \
  --img_size=224 \
  --train_csv='' \
  --test_csv='' \
  --optimizer='sgd' \
  --save_cm \
  --personal_eval \



