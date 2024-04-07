CUDA=$1
LOGGER=$2

HF_MODEL="FacebookAI/roberta-base"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL_LIGHT="8dc593b1b1a44946a004be496c5fb242"
TRIVIAL_LIGHT_MIXED="5bf2e076fe5c46f29b6dba6839670cc8"
ADVANCED_MIXED="335b5076295c408fbb02a67cc6a09705"
TRIVIAL_HEAVY_MIXED="27080f80029349e4a91d506060f2584b"

EVALUATE () {
    bash $SCRIPT \
        $TRIVIAL_LIGHT \
        eval-$1-trivial-light \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "avg"
        

    bash $SCRIPT \
        $TRIVIAL_LIGHT_MIXED \
        eval-$1-trivial-light-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "avg"

    bash $SCRIPT \
        $ADVANCED_MIXED \
        eval-$1-advanced-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "avg"
    
    bash $SCRIPT \
        $TRIVIAL_HEAVY_MIXED \
        eval-$1-trivial-heavy-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "avg"
}

# ==== RUN ====

EVALUATE "robert" "benchmarks-roberta" "multiclass"
