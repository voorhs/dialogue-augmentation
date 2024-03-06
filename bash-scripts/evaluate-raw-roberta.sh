CUDA=$1
LOGGER=$2

HF_MODEL="FacebookAI/roberta-base"
SCRIPT="bash-scripts/evaluate-raw.sh"

EVALUATE () {
    bash $SCRIPT \
        eval-raw-roberta-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER
}

# ==== RUN ====

EVALUATE "one-domain" "benchmarks-roberta" "multiclass"

# EVALUATE "multi-domain" "benchmarks-md-roberta" "multilabel"
