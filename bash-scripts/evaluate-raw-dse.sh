CUDA=$1
LOGGER=$2

HF_MODEL="aws-ai/dse-bert-base"
SCRIPT="bash-scripts/evaluate-raw.sh"

EVALUATE () {
    bash $SCRIPT \
        eval-raw-dse-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER
}

# ==== RUN ====

EVALUATE "one-domain" "benchmarks-dse" "multiclass"

# EVALUATE "multi-domain" "benchmarks-md-bert" "multilabel"
