INIT_FROM=$1
NAME=$2
HF_MODEL=$3
BENCHMARKS=$4
CUDA=$5
LOGGER=$6
MULTILABEL=$7
POOLING=$8

source .venv/bin/activate

python3 train_baseline_dialogue_encoder.py \
--hf-model $HF_MODEL \
--contrastive-path data/train-bert/trivial/ \
--multiwoz-path data/$BENCHMARKS/multiwoz \
--bitod-path data/$BENCHMARKS/bitod \
--sgd-path data/$BENCHMARKS/sgd \
--cuda $CUDA \
--logger $LOGGER \
--pooling $POOLING \
--batch-size 32 \
--finetune-layers 0 \
--name $NAME \
--validate \
--init-from $INIT_FROM \
--n-workers 4 \
--benchmark $MULTILABEL
