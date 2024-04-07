CUDA=$1
LOGGER=$2

HF_MODEL="FacebookAI/roberta-base"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL="e4fa2d98923e4c8ca4fbe20c7ccc4c67"
ADVANCED="849fa15859484683a4341837e7d1dd1d"
CRAZY="523da04c3c5a41379acc298ee09c7a0a"
HALVES="caee9e3358d9476fa0f6d1504740239e"
HALVES_DROPOUT="1509ea32ae0a4fe29a0048a57954fcfc"

EVALUATE () {
    bash $SCRIPT \
        $HALVES_DROPOUT \
        eval-$1-halves-dropout \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "avg"

    bash $SCRIPT \
        $HALVES \
        eval-$1-halves \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"


    bash $SCRIPT \
        $CRAZY \
        eval-$1-crazy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $ADVANCED \
        eval-$1-advanced-heavy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"

    bash $SCRIPT \
        $TRIVIAL \
        eval-$1-trivial-heavy \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \
        "cls"
}

# ==== RUN ====

EVALUATE "robert" "benchmarks-roberta" "multiclass"
