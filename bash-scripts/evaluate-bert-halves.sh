CUDA=$1
LOGGER=$2

HF_MODEL="google-bert/bert-base-uncased"
SCRIPT="bash-scripts/collect-checkpoints.sh"

HALVES_DROPOUT="e1d6e434bef840b78509477dbc8505d3"

# ==== RUN ====

EVALUATE  

bash $SCRIPT \
    $HALVES_DROPOUT \
    eval-bert-halves \
    $HF_MODEL \
    "benchmarks-bert" \
    $CUDA \
    "multiclass" \
    $LOGGER \ 
