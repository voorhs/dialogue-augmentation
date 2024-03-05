CUDA=$1
LOGGER=$2

HF_MODEL="Shitao/RetroMAE"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL_FAIR="408af9ac5d114a9da1fa007277343f15"
ADVANCED_FAIR="e2de548066f14753bdf9a646e992c02f"
CRAZY_FAIR="ca66a8cd07c4418daea805093d00cdd3"

TRIVIAL_UNFAIR="7b2d20d36b6c41778ac6ac44c13cac66"
ADVANCED_UNFAIR="abcf0e71cb7347db8726cf2f119939bf"
CRAZY_UNFAIR="739a35638b8e4031b747c99954d952f8"

EVALUATE_FAIR () {
    bash $SCRIPT \
        $TRIVIAL_FAIR \
        eval-retromae-trivial-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $ADVANCED_FAIR \
        eval-retromae-adanced-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $CRAZY_FAIR \
        eval-retromae-crazy-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 
}


EVALUATE_UNFAIR () {
    bash $SCRIPT \
        $TRIVIAL_UNFAIR \
        eval-retromae-trivial-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $ADVANCED_UNFAIR \
        eval-retromae-adanced-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $ADVANCED_UNFAIR \
        eval-retromae-crazy-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 
}

# ==== RUN ====

EVALUATE_FAIR "-one-domain" "benchmarks-retromae" false
EVALUATE_UNFAIR "-one-domain" "benchmarks-retromae" false

EVALUATE_FAIR "-multi-domain" "benchmarks-md-retromae" true
EVALUATE_UNFAIR "-multi-domain" "benchmarks-md-retromae" true
