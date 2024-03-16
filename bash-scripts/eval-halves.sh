CUDA=$1
LOGGER=$2
SCRIPT="bash-scripts/collect-checkpoints.sh"

# === BERT ===

HF_MODEL="google-bert/bert-base-uncased"
HALVES_DROPOUT="e1d6e434bef840b78509477dbc8505d3"

bash $SCRIPT \
    $HALVES_DROPOUT \
    eval-bert-halves \
    $HF_MODEL \
    "benchmarks-bert" \
    $CUDA \
    "multiclass" \
    $LOGGER \ 

# === RoBERTA ===

HF_MODEL="FacebookAI/roberta-base"
HALVES_DROPOUT="1509ea32ae0a4fe29a0048a57954fcfc"

bash $SCRIPT \
    $TRIVIAL_FAIR \
    eval-roberta-halves \
    $HF_MODEL \
    "benchmarks-roberta" \
    $CUDA \
    "multiclass" \
    $LOGGER \ 

# == RetroMAE ===

HF_MODEL="Shitao/RetroMAE"
MODEL_PATH="30b0491f250845eb89223c9abb4ef5bb"

bash $SCRIPT \
    $MODEL_PATH \
    eval-retromae-halves \
    $HF_MODEL \
    "benchmarks-retromae" \
    $CUDA \
    "multiclass" \
    $LOGGER
