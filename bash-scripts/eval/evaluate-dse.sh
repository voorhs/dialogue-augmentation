CUDA=$1
LOGGER=$2

HF_MODEL="aws-ai/dse-bert-base"
SCRIPT="bash-scripts/collect-checkpoints.sh"

CLS="edd298120d9b4fe6bab64dc9f5b3d939"
AVG="e82a8460c4c247958b0c5ec5a0bcfcf9"

bash $SCRIPT \
    $CLS \
    eval-dse-advanced-dse-light-mixed-cls \
    $HF_MODEL \
    benchmarks-dse \
    $CUDA \
    multiclass \
    $LOGGER \
    "cls"

bash $SCRIPT \
    $AVG \
    eval-dse-advanced-dse-light-mixed-avg \
    $HF_MODEL \
    benchmarks-dse \
    $CUDA \
    multiclass \
    $LOGGER \
    "avg"