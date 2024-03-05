CUDA=$1
LOGGER=$2

HF_MODEL="google-bert/bert-base-uncased"
SCRIPT="bash-scripts/collect-checkpoints.sh"

TRIVIAL_FAIR="ba6388e710684d94af81ebab88c2ff5a"
ADVANCED_FAIR="0d0b24e2d3864cf8a6c8b7d5e15b1c83"
CRAZY_FAIR="d7e24b7b434e486d866f35d70ac0c503"

TRIVIAL_UNFAIR="32118387351f4edf85d85582058e99ca"
ADVANCED_UNFAIR="38b54997a8704482b53f3935df110162"
CRAZY_UNFAIR="07045f26f4554abc98058a25c1114f71"

EVALUATE_FAIR () {
    bash $SCRIPT \
        $TRIVIAL_FAIR \
        eval-bert-trivial-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $LOGGER

    bash $SCRIPT \
        $ADVANCED_FAIR \
        eval-bert-adanced-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $LOGGER

    bash $SCRIPT \
        $CRAZY_FAIR \
        eval-bert-crazy-fair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $LOGGER
}


EVALUATE_UNFAIR () {
    bash $SCRIPT \
        $TRIVIAL_UNFAIR \
        eval-bert-trivial-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $LOGGER

    bash $SCRIPT \
        $ADVANCED_UNFAIR \
        eval-bert-adanced-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $LOGGER

    bash $SCRIPT \
        $ADVANCED_UNFAIR \
        eval-bert-crazy-unfair"$1" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $LOGGER
}

# ==== RUN ====

EVALUATE_FAIR "-one-domain" "benchmarks-bert"
EVALUATE_UNFAIR "-one-domain" "benchmarks-bert"

EVALUATE_FAIR "-multi-domain" "benchmarks-md-bert"
EVALUATE_UNFAIR "-multi-domain" "benchmarks-md-bert"
