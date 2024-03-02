source .venv/bin/activate

# bert
python3 train_baseline_dialogue_encoder.py \
--hf-model google-bert/bert-base-uncased \
--contrastive-path data/train-bert-base-uncased/trivial/ \
--multiwoz-path data/benchmarks-bert-base-uncased/multiwoz \
--bitod-path data/benchmarks-bert-base-uncased/bitod \
--sgd-path data/benchmarks-bert-base-uncased/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name bert-trivial

python3 train_baseline_dialogue_encoder.py \
--hf-model google-bert/bert-base-uncased \
--contrastive-path data/train-bert-base-uncased/advanced/ \
--multiwoz-path data/benchmarks-bert-base-uncased/multiwoz \
--bitod-path data/benchmarks-bert-base-uncased/bitod \
--sgd-path data/benchmarks-bert-base-uncased/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name bert-advanced

python3 train_baseline_dialogue_encoder.py \
--hf-model google-bert/bert-base-uncased \
--contrastive-path data/train-bert-base-uncased/crazy/ \
--multiwoz-path data/benchmarks-bert-base-uncased/multiwoz \
--bitod-path data/benchmarks-bert-base-uncased/bitod \
--sgd-path data/benchmarks-bert-base-uncased/sgd \
--cuda "0,1" \
--logger comet \
--pooling cls \
--batch-size 128 \
--finetune-layers 3 \
--loss contrastive_symmetric \
--name bert-crazy

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
--name roberta-trivial

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
--name roberta-advanced

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
--name roberta-crazy

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
--name retromae-trivial

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
--name retromae-advanced

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
--name retromae-crazy
