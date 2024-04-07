#!/bin/bash

CKPT_CODE=$1
NAME=$2
HF_MODEL=$3
BENCHMARKS=$4
CUDA=$5
LOGGER=$7
MULTILABEL=$6
POOLING=$8

for CKPT in ~/dialogue-augmentation/logs/comet/dialogue-encoder/$CKPT_CODE/checkpoints/*.ckpt; do
    CKPT_NAME=$(basename $CKPT)
    EXP_NAME="${NAME}-${CKPT_NAME}"
    bash ~/dialogue-augmentation/bash-scripts/evaluate.sh $CKPT $EXP_NAME $HF_MODEL $BENCHMARKS $CUDA $LOGGER $MULTILABEL $POOLING
done
