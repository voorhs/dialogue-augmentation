CUDA=$1
LOGGER=$2

HF_MODEL="FacebookAI/roberta-base"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL_FAIR="e4fa2d98923e4c8ca4fbe20c7ccc4c67"
ADVANCED_FAIR="849fa15859484683a4341837e7d1dd1d"
CRAZY_FAIR="523da04c3c5a41379acc298ee09c7a0a"

TRIVIAL_UNFAIR="115f79b06edc485e9477cffdf52d3293"
ADVANCED_UNFAIR="e53f956ef0b5438094847ad06d2d374a"
CRAZY_UNFAIR="ecf6a3d7ae5a4293bd12171794b09a0f"

EVALUATE_FAIR () {
    bash $SCRIPT \
        $TRIVIAL_FAIR \
        eval-roberta-trivial-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $ADVANCED_FAIR \
        eval-roberta-adanced-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $CRAZY_FAIR \
        eval-roberta-crazy-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 
}


EVALUATE_UNFAIR () {
    bash $SCRIPT \
        $TRIVIAL_UNFAIR \
        eval-roberta-trivial-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $ADVANCED_UNFAIR \
        eval-roberta-adanced-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 

    bash $SCRIPT \
        $ADVANCED_UNFAIR \
        eval-roberta-crazy-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 
}

# ==== RUN ====

EVALUATE_FAIR "-one-domain" "benchmarks-roberta" "multiclass"
EVALUATE_UNFAIR "-one-domain" "benchmarks-roberta" "multilabel"

EVALUATE_FAIR "-multi-domain" "benchmarks-md-roberta" "multilabel"
EVALUATE_UNFAIR "-multi-domain" "benchmarks-md-roberta" "multilabel"
