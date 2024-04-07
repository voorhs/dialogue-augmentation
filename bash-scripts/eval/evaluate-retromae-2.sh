CUDA=$1
LOGGER=$2

HF_MODEL="Shitao/RetroMAE"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL_LIGHT="ac2cd44f7c13474e9d68dee28f3ba2c8"
TRIVIAL_LIGHT_MIXED="fa671f2d23bd4b26a0dd3230d6a5b39b"
ADVANCED_MIXED="622b1b8534a54f4fa400f1a5264b62e1"
TRIVIAL_HEAVY_MIXED="4df5a5164cfb478692a7a6ca6ebf153b"

EVALUATE () {
   bash $SCRIPT \
        $TRIVIAL_LIGHT \
        eval-$1-trivial-light \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $TRIVIAL_LIGHT_MIXED \
        eval-$1-trivial-light-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $ADVANCED_MIXED \
        eval-$1-advanced-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"
    
    bash $SCRIPT \
        $TRIVIAL_HEAVY_MIXED \
        eval-$1-trivial-heavy-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"
}

# ==== RUN ====

EVALUATE "retromae" "benchmarks-retromae" "multiclass"
