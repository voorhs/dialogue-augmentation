CUDA=$1
LOGGER=$2

HF_MODEL="FacebookAI/roberta-base"
SCRIPT="bash-scripts/collect-checkpoints.sh"

HALVES_DROPOUT="1509ea32ae0a4fe29a0048a57954fcfc"


bash $SCRIPT \
    $TRIVIAL_FAIR \
    eval-roberta-trivial-halves \
    $HF_MODEL \
    "benchmarks-roberta" \
    $CUDA \
    "multiclass" \
    $LOGGER \ 
