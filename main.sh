# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/home/kishida_local/ILSVRC/Data/CLS-LOC/train"
ARCH="alexnet"
LR=0.05
WD=-5
WORKERS=12
EXP="/home/kishida_local/test/exp"
PYTHON="/home/kishida_local/.pyenv/shims/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --sobel --verbose --workers ${WORKERS}
