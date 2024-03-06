NAME=$1
HF_MODEL=$2
BENCHMARKS=$3
CUDA=$4
MULTILABEL=$5
LOGGER=$6

source .venv/bin/activate

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
--n-workers 4 \
--benchmark $MULTILABEL
