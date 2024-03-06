CUDA=$1
LOGGER=$2

HF_MODEL="FacebookAI/roberta-base"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL_FAIR="caee9e3358d9476fa0f6d1504740239e"
ADVANCED_FAIR="f60540d763ff4260919483147d8d2e28"
CRAZY_FAIR="8a80bf7248b14f23ba6640fae2d4a9b6"


EVALUATE () {
    bash $SCRIPT \
        $TRIVIAL_FAIR \
        eval-roberta-trivial-halves-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $ADVANCED_FAIR \
        eval-roberta-adanced-halves-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $CRAZY_FAIR \
        eval-roberta-crazy-halves-"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER
}


# ==== RUN ====

EVALUATE "one-domain" "benchmarks-roberta" "multiclass"

# EVALUATE "multi-domain" "benchmarks-md-roberta" "multilabel"
