CUDA=$1
LOGGER=$2

HF_MODEL="google-bert/bert-base-uncased"
SCRIPT="bash-scripts/collect-checkpoints.sh"

ADVANCED_LIGHT_MIXED="6d6dcd8d53c343b2b2ae88dc140af753"
ADVANCED_LIGHT_DSE_MIXED="5058e4ab7f0e4953818da04bdb1d81e6"

EVALUATE () {
    bash $SCRIPT \
        $ADVANCED_LIGHT_MIXED \
        eval-$1-advanced-light-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $ADVANCED_LIGHT_DSE_MIXED \
        eval-$1-advanced-light-dse-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        cls
}

# ==== RUN ====

EVALUATE "bert" "benchmarks-bert" "multiclass"
