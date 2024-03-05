CUDA=$1
LOGGER=$2
N_CLASSES=$3

HF_MODEL="google-bert/bert-base-uncased"
SCRIPT="bash-scripts/collect-checkpoints-finetune.sh"

TRIVIAL_FAIR="ba6388e710684d94af81ebab88c2ff5a"
ADVANCED_FAIR="0d0b24e2d3864cf8a6c8b7d5e15b1c83"
CRAZY_FAIR="d7e24b7b434e486d866f35d70ac0c503"

TRIVIAL_UNFAIR="32118387351f4edf85d85582058e99ca"
ADVANCED_UNFAIR="38b54997a8704482b53f3935df110162"
CRAZY_UNFAIR="07045f26f4554abc98058a25c1114f71"

EVALUATE_FAIR () {
    bash $SCRIPT \
        $TRIVIAL_FAIR \
        finetune-bert-trivial-fair"$1"-"$2" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 
        $N_CLASSES \

    bash $SCRIPT \
        $ADVANCED_FAIR \
        finetune-bert-adanced-fair"$1"-"$2" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 
        $N_CLASSES \

    bash $SCRIPT \
        $CRAZY_FAIR \
        finetune-bert-crazy-fair"$1"-"$2" \
        $HF_MODEL \
        $2 \
        $CUDA \
        $3 \
        $LOGGER \ 
        $N_CLASSES \
}


# EVALUATE_UNFAIR () {
#     bash $SCRIPT \
#         $TRIVIAL_UNFAIR \
#         finetune-bert-trivial-unfair"$1"-"$2" \
#         $HF_MODEL \
#         $2 \
#         $CUDA \
#         $3 \
#         $LOGGER \ 
#         $N_CLASSES \

#     bash $SCRIPT \
#         $ADVANCED_UNFAIR \
#         finetune-bert-adanced-unfair"$1"-"$2" \
#         $HF_MODEL \
#         $2 \
#         $CUDA \
#         $3 \
#         $LOGGER \ 
#         $N_CLASSES \

#     bash $SCRIPT \
#         $ADVANCED_UNFAIR \
#         finetune-bert-crazy-unfair"$1"-"$2" \
#         $HF_MODEL \
#         $2 \
#         $CUDA \
#         $3 \
#         $LOGGER \ 
#         $N_CLASSES \
# }

# ==== RUN ====

# EVALUATE_FAIR "-one-domain" "data/benchmarks-bert/multiwoz" "multiclass"
# EVALUATE_FAIR "-one-domain" "data/benchmarks-bert/bitod" "multiclass"
# EVALUATE_FAIR "-one-domain" "data/benchmarks-bert/sgd" "multiclass"
# EVALUATE_UNFAIR "-one-domain" "data/benchmarks-bert/multiwoz" "multiclass"
# EVALUATE_UNFAIR "-one-domain" "data/benchmarks-bert/bitod" "multiclass"
# EVALUATE_UNFAIR "-one-domain" "data/benchmarks-bert/sgd" "multiclass"

# EVALUATE_FAIR "-multi-domain" "data/benchmarks-md-bert/multiwoz" "multilabel"
# EVALUATE_FAIR "-multi-domain" "data/benchmarks-md-bert/bitod" "multilabel"
# EVALUATE_FAIR "-multi-domain" "data/benchmarks-md-bert/sgd" "multilabel"
# EVALUATE_UNFAIR "-multi-domain" "data/benchmarks-md-bert/multiwoz" "multilabel"
# EVALUATE_UNFAIR "-multi-domain" "data/benchmarks-md-bert/bitod" "multilabel"
# EVALUATE_UNFAIR "-multi-domain" "data/benchmarks-md-bert/sgd" "multilabel"
