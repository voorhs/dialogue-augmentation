CUDA=$1
LOGGER=$2

HF_MODEL="Shitao/RetroMAE"
SCRIPT="bash-scripts/collect-checkpoints.sh"

ADVANCED_LIGHT_MIXED="b5726bb95a7844c1a5bc54bc1678fd66"
ADVANCED_LIGHT_DSE_MIXED="79a67d9022644dd788ff237a4e5da0fa"

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

EVALUATE "retromae" "benchmarks-bert" "multiclass"
