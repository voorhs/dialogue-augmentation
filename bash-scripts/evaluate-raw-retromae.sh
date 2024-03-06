CUDA=$1
LOGGER=$2

HF_MODEL="Shitao/RetroMAE"
SCRIPT="bash-scripts/evaluate-raw.sh"

EVALUATE () {
    bash $SCRIPT \
        eval-raw-retromae-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER
}

# ==== RUN ====

EVALUATE "one-domain" "benchmarks-retromae" "multiclass"

# EVALUATE "multi-domain" "benchmarks-md-retromae" "multilabel"
