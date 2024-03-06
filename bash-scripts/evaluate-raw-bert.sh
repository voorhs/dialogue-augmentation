CUDA=$1
LOGGER=$2

HF_MODEL="google-bert/bert-base-uncased"
SCRIPT="bash-scripts/evaluate-raw.sh"

EVALUATE () {
    bash $SCRIPT \
        eval-raw-bert-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER
}

# ==== RUN ====

EVALUATE "one-domain" "benchmarks-bert" "multiclass"

EVALUATE "multi-domain" "benchmarks-md-bert" "multilabel"
