INIT_FROM=$1
NAME=$2
HF_MODEL=$3
BENCHMARKS=$4
CUDA=$5
LOGGER=$6

source .venv/bin/activate

# bert
python3 train_baseline_dialogue_encoder.py \
--hf-model $HF_MODEL \
--contrastive-path data/train-bert/trivial/ \
--multiwoz-path data/$BENCHMARKS/multiwoz \
--bitod-path data/$BENCHMARKS/bitod \
--sgd-path data/$BENCHMARKS/sgd \
--cuda $CUDA \
--logger $LOGGER \
--pooling cls \
--batch-size 128 \
--finetune-layers 0 \
--name $NAME \
--validate \
--init-from $INIT_FROM \
--n-workers 4


# roberta
# python3 train_baseline_dialogue_encoder.py \
# --hf-model FacebookAI/roberta-base \
# --contrastive-path data/train-roberta/trivial/ \
# --multiwoz-path data/benchmarks-roberta/multiwoz \
# --bitod-path data/benchmarks-roberta/bitod \
# --sgd-path data/benchmarks-roberta/sgd \
# --cuda "0,1" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name roberta-trivial-fair-eval

# python3 train_baseline_dialogue_encoder.py \
# --hf-model FacebookAI/roberta-base \
# --contrastive-path data/train-roberta/advanced/ \
# --multiwoz-path data/benchmarks-roberta/multiwoz \
# --bitod-path data/benchmarks-roberta/bitod \
# --sgd-path data/benchmarks-roberta/sgd \
# --cuda "0,1" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name roberta-advanced-fair-eval

# python3 train_baseline_dialogue_encoder.py \
# --hf-model FacebookAI/roberta-base \
# --contrastive-path data/train-roberta/crazy/ \
# --multiwoz-path data/benchmarks-roberta/multiwoz \
# --bitod-path data/benchmarks-roberta/bitod \
# --sgd-path data/benchmarks-roberta/sgd \
# --cuda "0,1" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name roberta-crazy-fair-eval

# # retromae
# python3 train_baseline_dialogue_encoder.py \
# --hf-model Shitao/RetroMAE \
# --contrastive-path data/train-retromae/trivial/ \
# --multiwoz-path data/benchmarks-retromae/multiwoz \
# --bitod-path data/benchmarks-retromae/bitod \
# --sgd-path data/benchmarks-retromae/sgd \
# --cuda "0,1" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name retromae-trivial-fair-eval

# python3 train_baseline_dialogue_encoder.py \
# --hf-model Shitao/RetroMAE \
# --contrastive-path data/train-retromae/advanced/ \
# --multiwoz-path data/benchmarks-retromae/multiwoz \
# --bitod-path data/benchmarks-retromae/bitod \
# --sgd-path data/benchmarks-retromae/sgd \
# --cuda "0,1" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name retromae-advanced-fair-eval

# python3 train_baseline_dialogue_encoder.py \
# --hf-model Shitao/RetroMAE \
# --contrastive-path data/train-retromae/crazy/ \
# --multiwoz-path data/benchmarks-retromae/multiwoz \
# --bitod-path data/benchmarks-retromae/bitod \
# --sgd-path data/benchmarks-retromae/sgd \
# --cuda "0,1" \
# --logger comet \
# --pooling cls \
# --batch-size 128 \
# --finetune-layers 3 \
# --loss contrastive_symmetric \
# --name retromae-crazy-fair-eval
