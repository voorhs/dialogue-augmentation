CUDA=$1
LOGGER=$2

HF_MODEL="google-bert/bert-base-uncased"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL_LIGHT="d04c1de3f68440899e4463c6b467bde1"
TRIVIAL_LIGHT_MIXED="8d19505a49b842ba8b2b72717eb0c2ff"
ADVANCED_MIXED="0e19c7c3f39c4f238bb6293b771d17fd"
TRIVIAL_HEAVY_MIXED="b19de997b67b4c21baa61ce8fe4722a9"

EVALUATE () {
    bash $SCRIPT \
        $TRIVIAL_LIGHT \
        eval-$1-trivial-light \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls" \

    bash $SCRIPT \
        $TRIVIAL_LIGHT_MIXED \
        eval-$1-trivial-light-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        cls \

    bash $SCRIPT \
        $ADVANCED_MIXED \
        eval-$1-advanced-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        cls
    
    bash $SCRIPT \
        $TRIVIAL_HEAVY_MIXED \
        eval-$1-trivial-heavy-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        cls
}

# ==== RUN ====

EVALUATE "bert" "benchmarks-bert" "multiclass"
