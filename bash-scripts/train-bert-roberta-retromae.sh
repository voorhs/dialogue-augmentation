source .venv/bin/activate

# bert
python3 train_baseline_dialogue_encoder.py \
--hf-model google-bert/bert-base-uncased \
--contrastive-path data/train-bert/trivial/ \
--multiwoz-path data/benchmarks-bert/multiwoz \
--bitod-path data/benchmarks-bert/bitod \
--sgd-path data/benchmarks-bert/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name bert-trivial-fair

python3 train_baseline_dialogue_encoder.py \
--hf-model google-bert/bert-base-uncased \
--contrastive-path data/train-bert/advanced/ \
--multiwoz-path data/benchmarks-bert/multiwoz \
--bitod-path data/benchmarks-bert/bitod \
--sgd-path data/benchmarks-bert/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name bert-advanced-fair

python3 train_baseline_dialogue_encoder.py \
--hf-model google-bert/bert-base-uncased \
--contrastive-path data/train-bert/crazy/ \
--multiwoz-path data/benchmarks-bert/multiwoz \
--bitod-path data/benchmarks-bert/bitod \
--sgd-path data/benchmarks-bert/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name bert-crazy-fair

# roberta
python3 train_baseline_dialogue_encoder.py \
--hf-model FacebookAI/roberta-base \
--contrastive-path data/train-roberta/trivial/ \
--multiwoz-path data/benchmarks-roberta/multiwoz \
--bitod-path data/benchmarks-roberta/bitod \
--sgd-path data/benchmarks-roberta/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name roberta-trivial-fair

python3 train_baseline_dialogue_encoder.py \
--hf-model FacebookAI/roberta-base \
--contrastive-path data/train-roberta/advanced/ \
--multiwoz-path data/benchmarks-roberta/multiwoz \
--bitod-path data/benchmarks-roberta/bitod \
--sgd-path data/benchmarks-roberta/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name roberta-advanced-fair

python3 train_baseline_dialogue_encoder.py \
--hf-model FacebookAI/roberta-base \
--contrastive-path data/train-roberta/crazy/ \
--multiwoz-path data/benchmarks-roberta/multiwoz \
--bitod-path data/benchmarks-roberta/bitod \
--sgd-path data/benchmarks-roberta/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name roberta-crazy-fair

# retromae
python3 train_baseline_dialogue_encoder.py \
--hf-model Shitao/RetroMAE \
--contrastive-path data/train-retromae/trivial/ \
--multiwoz-path data/benchmarks-retromae/multiwoz \
--bitod-path data/benchmarks-retromae/bitod \
--sgd-path data/benchmarks-retromae/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name retromae-trivial-fair

python3 train_baseline_dialogue_encoder.py \
--hf-model Shitao/RetroMAE \
--contrastive-path data/train-retromae/advanced/ \
--multiwoz-path data/benchmarks-retromae/multiwoz \
--bitod-path data/benchmarks-retromae/bitod \
--sgd-path data/benchmarks-retromae/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name retromae-advanced-fair

python3 train_baseline_dialogue_encoder.py \
--hf-model Shitao/RetroMAE \
--contrastive-path data/train-retromae/crazy/ \
--multiwoz-path data/benchmarks-retromae/multiwoz \
--bitod-path data/benchmarks-retromae/bitod \
--sgd-path data/benchmarks-retromae/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name retromae-crazy-fair
