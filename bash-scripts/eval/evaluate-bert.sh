CUDA=$1
LOGGER=$2

HF_MODEL="google-bert/bert-base-uncased"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL="ba6388e710684d94af81ebab88c2ff5a"
ADVANCED="0d0b24e2d3864cf8a6c8b7d5e15b1c83"
CRAZY="d7e24b7b434e486d866f35d70ac0c503"
HALVES="dd2c00a94bab43d5a85588782d1c16a9"
HALVES_DROPOUT="e1d6e434bef840b78509477dbc8505d3"

EVALUATE () {
    # bash $SCRIPT \
    #     $HALVES_DROPOUT \
    #     eval-$1-halves-dropout \
    #     $HF_MODEL \
    #     $2 \
    #     $CUDA \
    #     $3 \
    #     $LOGGER \
    #     cls

    bash $SCRIPT \
        $HALVES \
        eval-$1-halves \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        cls

    bash $SCRIPT \
        $CRAZY \
        eval-$1-crazy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        cls

    bash $SCRIPT \
        $ADVANCED \
        eval-$1-advanced-heavy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        cls

    bash $SCRIPT \
        $TRIVIAL \
        eval-$1-trivial-heavy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        cls
}

# ==== RUN ====

EVALUATE "bert" "benchmarks-bert" "multiclass"
