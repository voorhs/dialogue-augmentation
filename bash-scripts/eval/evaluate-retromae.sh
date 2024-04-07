CUDA=$1
LOGGER=$2

HF_MODEL="Shitao/RetroMAE"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL="408af9ac5d114a9da1fa007277343f15"
ADVANCED="e2de548066f14753bdf9a646e992c02f"
CRAZY="ca66a8cd07c4418daea805093d00cdd3"
HALVES="e74b5f8cfd5849ee88715b784f731b3c"
HALVES_DROPOUT="30b0491f250845eb89223c9abb4ef5bb"

EVALUATE () {
    bash $SCRIPT \
        $HALVES_DROPOUT \
        eval-$1-halves-dropout \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $HALVES \
        eval-$1-halves \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $CRAZY \
        eval-$1-crazy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $ADVANCED \
        eval-$1-advanced-heavy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $TRIVIAL \
        eval-$1-trivial-heavy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"
}

# ==== RUN ====

EVALUATE "retromae" "benchmarks-retromae" "multiclass"
