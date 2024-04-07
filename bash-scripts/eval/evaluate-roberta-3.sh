CUDA=$1
LOGGER=$2

HF_MODEL="FacebookAI/roberta-base"
SCRIPT="bash-scripts/collect-checkpoints.sh"

ADVANCED_LIGHT_MIXED="88866b87989c49fbb8e01cd5b7d5ae60"
ADVANCED_LIGHT_DSE_MIXED="ab2d2970e5f240ddbe050e1fdb589d29"

EVALUATE () {
    bash $SCRIPT \
        $ADVANCED_LIGHT_MIXED \
        eval-$1-advanced-light-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "avg"

    bash $SCRIPT \
        $ADVANCED_LIGHT_DSE_MIXED \
        eval-$1-advanced-light-dse-mixed \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        avg
}

# ==== RUN ====

EVALUATE "roberta" "benchmarks-bert" "multiclass"
