#!/bin/bash

CUDA=$1
LOGGER=$2

HF_MODEL="Shitao/RetroMAE"
SCRIPT="bash-scripts/collect-checkpoints.sh"
MODEL_PATH="30b0491f250845eb89223c9abb4ef5bb"

bash $SCRIPT \
    $MODEL_PATH \
    eval-retromae-halves \
    $HF_MODEL \
    "benchmarks-retromae" \
    $CUDA \
    "multiclass" \
    $LOGGER