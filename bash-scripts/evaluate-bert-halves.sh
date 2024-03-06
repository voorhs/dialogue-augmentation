CUDA=$1
LOGGER=$2

HF_MODEL="google-bert/bert-base-uncased"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL_FAIR="dd2c00a94bab43d5a85588782d1c16a9"
ADVANCED_FAIR="c1a1debf410f41d4baece64c36b322d8"
CRAZY_FAIR="40481ab4834f40a5bc2c0df4862d3adb"


EVALUATE () {
    bash $SCRIPT \
        $TRIVIAL_FAIR \
        eval-bert-trivial-halves-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $ADVANCED_FAIR \
        eval-bert-adanced-halves-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $CRAZY_FAIR \
        eval-bert-crazy-halves-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER
}


# ==== RUN ====

EVALUATE "one-domain" "benchmarks-bert" "multiclass"

EVALUATE "multi-domain" "benchmarks-md-bert" "multilabel"
