# Dialogue Augmentation

Скрипты:
- скрипты для DGAC кластеризации:
    - `parse_multiwoz.py`
    - `sentence_encoding.py`
    - `dgac_clustering.py`
- реализованные аугментации:
    - `my_augmentations.py`
    - `augmentation_development.ipynb`
- скрипты для анализа intent similarity и edit similarity:
    - `grid-search.py`
    - `similarity_funtions.py`
    - `visualization_utis.py`
    - `similarity_analysis.ipynb`

## Как сделать датасет

Контрастивный датасет для трейна диалогового энкодера:
- `make_source_dataset.py` => `data/source/train/`, датасет из диалогов
- `filter_dataset_by_length.py` => `data/misc/original-truncated/`, датасет с нанами на месте слишком длинных диалогов (`mode='null'`)
- `make_augmentations.py` => `augmented/`, всевозможные аугментации трейна
- `make_contrastive_dialogue_train_data.py` => `data/train/dialogue-encoder/contrastive/`

Мультивоз датасет для service clf:
- `make_multiwoz_dataset.py` => `data/misc/multiwoz22/`
- `filter_dataset_by_length.py` => `data/train/dialogue-encoder/multiwoz22`, датасет с без слишком длинных диалогов (`mode='drop'`)



## Текущие мысли

- `dice`-близость над `bag of nodes` хороша, но только в случае устойчивой кластеризации
    - было [A] [label: 5] [name: cambridge, leaving, need]  My departure site is ~~Cambridge~~ please.
    - стало [A] [label: 13] [name: centre, looking, town] My departure site is **Bend** please.
    - было [A] [label: 14] [name: food, italian, restaurant] I would like modern ~~European~~ food.
    - стало [A] [label: 0] [name: food, looking, restaurant] I would like modern **Asian** food.
    - т.е. из-за смещения в данных, возник отдельный кластер про кембридж/италию и это сделало кластеризацию не инвариантной к фактической информации, по-моему нехорошо, что кластер меняется при том же интенте
- `textattack` CLI
    - медленный
    - иногда непонятно какие модели он использует
    - принимает на вход только csv (хотелось бы json)
- с помощью замены синонимов можно генерить отрицательные примеры похожих диалогов, а с помощью добавления слов -- положительные
- LLM можно использовать для dialogue completion для генерации положительных примеров (убрать часть реплик и попросить бота заполнить)
- CLARE
    - вообще отличный метод для генерации положительных примеров, потому что он был создан как раз чтобы генерить положительные примеры, причем максимально нетривиальные (чтобы взломать модель)
    - но хотелось бы регулировать используемые трансформации, например, только вставки или любое другое подмножество, а не все сразу
    - но в textattack он слишком медленный: для обработки корпуса придется ждать несколько дней
    возможно нужно будет попробовать через api, мейби там есть какие-нибудь оптимизации или возможности для распараллеливания
    - в идеале самому написать на торче
- текущие итоги по `Inserter`
    - модели:
        - `microsoft/mpnet-base` отлично работает при `fill_utterance_level=2`, нетривиальные вставки, заодно даже немного взламывает кластеризацию
        - `xlnet-base-cased` и `xlnet-large-cased` вставляет одни предлоги и наречия, в целом глупее (причем при любой стратегии заполнения)
        - когда стоит использовать
            - mpnet: если не жалко времени (он раза в два медленее xlnet) и умещается в контекст из двух уттерансов 
            - xlnet: в остальных случаях
    - fraction:
        - если mpnet, то смело берем 0.5 и получаем отличные примеры
        - иначе 0.3
- как с помощью лламы аугментировать диалог:
    - маскировать уттерансы и просить лламу заполнить маски так, чтобы они подходили под контекст диалога
    - добавить маскированные уттерансы и попросить лламу заполнить их
    - попросить придумать начало/конец диалога
    - переписать диалог покороче/подлиннее
    - **ничего не удалось, не работает как часы: то лишние комментарии генерит, то нарушает json структуру**
    - **скорее всего нужно файнтюнить чтобы модель выдавала всегда одинаковый формат аутпута и лучше понимала что от нее требуют**
    - **наиболее продвинутой версией файнтюна мне видится комбинирование файнтюна с jsonformer**
- аугментация через шаффл:
    - выделять блоки (параграфы) и шаффлить их, а не отдельные предложения
    - например в мультивозе в одном диалоге часто речь сначала про ресторан а потом про такси, вот их можно шаффлить сто проц
    - то есть можно к примеру кластеризовать все уттерансы в датасете, и использовать разметку для сегментации диалога на смысловые части
    - можно посчитать попарные близости всех уттерансов в предложении и свапнуть пары самых близких
    - одну реплику состоящую из нескольких предложений можно разбить на несколько, не забыв указать говорящего, и наоборот

## Детектить корректный порядок реплик в диалоге с помощью NSP моделей

План работы:
- руками нагенерить шаффлы в диалогах: валидные и невалидные примеры
- собрать все эти NSP модели:
    - какие-нибудь с hf
    - собрать NSP датасет из диалоговых датасетов и обучить бейзлайн
- измерить NSP-скоры для сгенеренных шаффлов

Варианты NSP модели:
- спуллить ConveRT эмбеддинг уттерансов и обучить бустинг/ff-block/rnn на бинарную классификацию (correct/meaningless)
    - не забыть учесть спикера (обучить эмбеддинги для юзера и системы)
    - варианты пуллинга: min, max, avg, attention
- засунуть весь диалог в BiDeN и обучить CLS токен (или другая модель для dialogue modeling)
- два энкодера: один для предшествующего уттеранса, второй для последующего, максимизировать скалярное произведение если два уттеранса последовательные
    - энкодеры можно инициализировать из mpnet
    - можно файнтюнить энкодеры, а можно обучить проекторы
    - учесть контекст без дополнительного пересчета уттеранс-эмбеддингов можно так: считать выпуклую комбинацию текущего эмбеддинга и всех предыдущих
- **предсказывать ранг каждого уттеранса по данному диалогу: дать на вход уттерансы без информации об их порядке и попросить его восстановить (сделать лосс как в ListNet):**
    - каждый уттеранс кодируем мпнетом -> получили набор эмбеддингов
    - этот набор воспринимаем как токены, подаем на вход транформеру
    - хидден стейты с последнего слоя подаем в классификатор с одним выходом (это ранкер)
    - выходы для текущей последовательности подаем в софтмакс и воспринимаем полученные вероятности как ранги
    - применяем лосс из ListNet
    - сейчас надо реализовать
        - в уттеранс ранкере паддинг для батча из уттеранс эмбеддингов
        - в трансформере нулевой аттеншен для паддингов
        - прикинуть варианты распределения для `ranks_probs_true`
    - возможно нужно не прибавлять эмбеддинг говорящего а конкатенировать маленький кусочек

- кажется идеальным вариантом:
    - взять байден (или другую модель для dialogue modeling), зафайнтюнить его таким образом, чтобы на уровне диалога аттеншен был только между cls токенами каждого уттеранса, а внутри каждого уттеранса обычный аттеншен
    - закодить такую модель будет небыстро, поэтому сделать это стоит только после тестов над текущим вариантом уттеранс ранкера
    - можно и без байдена!
    - возьму MPNetModel из хф, подам в него тщательно задизайненую маску внимания, возможно в начало каждого уттеранса нужно добавить токен, отвечающий за говорящего

## Дальнейшие планы по переходу с json chunks на hf dataset

[x] реализовать natural join для аугментированных датасетов: `datasets.Dataset.to_parquet()` -> `pyarrow.Table.join()` -> `datasets.Dataset.from_parquet()` (получится hf-версия скрипта `make_contrastive_dialogue_train_data.py`)
[x] сделать hf-версию для `filter_dataset_by_length.py`
[x] сделать hf-версии для
    [x] `train_dialogue_encoder.py`
    [] ~~`train_listwise.py`~~
    [x] `train_pairwise.py`
[x] что все `train_*.py` работают:
    [x] `train_pairwise.py`
    [] `train_dialogue_encoder.py`
[x] мигрировать на кластер
[] обучить симметричный pairwise

## Виды контрастив лоссов

1d loss is equivalent to $p(y|x)p(x|y)$
$$
-\log{\exp\cos(x,y)\over\sum_{\tilde y}\exp\cos(x,\tilde y)}-\log{\exp\cos(x,y)\over\sum_{\tilde x}\exp\cos(\tilde x,y)}
$$

2d loss
$$
-\log{\exp\cos(x,y)\over\sum_{\tilde x,\tilde y}\exp\cos(\tilde x,\tilde y)}
$$

my proposal 0 (the first term from 1d loss)
$$
-\log{\exp\cos(x,y)\over\sum_{\tilde y}\exp\cos(x,\tilde y)}
$$

my proposal 1
$$
-\log{\exp\cos(x_i,y_i)\over\sum_{j=1,j\neq i}^B\exp\cos(x_i,y_j)+\exp\cos(x_j,y_i)}
$$

my proposal 2
$$
-\log\sigma(\cos(x_i,y_i))-\sum_{j=1,j\neq i}^B[\log\sigma(-\cos(x_i,y_j))+\log\sigma(-\cos(x_j,y_i))]
$$

add random swap between x and y to prevent learning grammatics

я вижу так:
- последним двум лоссам можно поставить в соответствие конкретное вероятностное распределение к которому ведет шаг оптимизации
- но последние два лосса возможно сильно делают объекты в батче зависимыми, при том что очев батч случайный

## Try Yourself

### Setup Environment

python 3.8.10
```bash
pip install -r requirements
```

you can install specific python version with [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#automatic-installer):
```bash
sudo apt update
sudo apt install \
    build-essential \
    curl \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev
curl https://pyenv.run | bash
```

Train dialogue encoder (domain benchmark as validation)
```bash
python3 train_baseline_dialogue_encoder.py \
--hf-model google-bert/bert-base-uncased \
--contrastive-path data/train-bert-base-uncased/trivial/ \
--multiwoz-path data/benchmarks-bert-base-uncased/multiwoz \
--bitod-path data/benchmarks-bert-base-uncased/bitod \
--sgd-path data/benchmarks-bert-base-uncased/sgd \
--cuda "0,1" \
--logger comet \
# --mode max \
--pooling cls \
# --metric-for-checkpoint multiwoz/logreg_accuracy \
--batch-size 128 \
--finetune-layers 3 \
--name trivial
```

Train pairwise model
```bash
python3 train_pairwise.py --cuda 0 --logger tb --mode max --metric-for-checkpoint val_loss --batch-size 256 --finetune-layers 3
```

### Algorithm

1. make_source_dataset.py
2. make_augmentations.py
3. make_contrastive.py
4. filter_dataset_by_length.py both for contrastive and multiwoz
5. train_dialogue_encoder.py

### Aug Sets

Набор 1 (trivial):

- back-translate
- insert
- replace

Набор 2 (advanced):

- back-translate
- insert
- replace
- shuffle
- prune

Набор 3 (crazy):

- back-translate
- insert
- replace
- shuffle
- prune
- back-translate-prune
- shuffle-insert
- shuffle-replace
- prune-insert
- prune-replace

## Веса

- Fair
    - BERT
        - trivial ba6388e710684d94af81ebab88c2ff5a
        - advanced 0d0b24e2d3864cf8a6c8b7d5e15b1c83
        - crazy d7e24b7b434e486d866f35d70ac0c503
    - RoBERTa
        - trivial e4fa2d98923e4c8ca4fbe20c7ccc4c67
        - advanced 849fa15859484683a4341837e7d1dd1d
        - crazy 523da04c3c5a41379acc298ee09c7a0a
    - RetroMAE
        - trivial 408af9ac5d114a9da1fa007277343f15
        - advanced e2de548066f14753bdf9a646e992c02f
        - crazy ca66a8cd07c4418daea805093d00cdd3
- Unair
    - BERT
        - trivial 32118387351f4edf85d85582058e99ca
        - advanced 38b54997a8704482b53f3935df110162
        - crazy 07045f26f4554abc98058a25c1114f71
    - RoBERTa
        - trivial 115f79b06edc485e9477cffdf52d3293
        - advanced e53f956ef0b5438094847ad06d2d374a
        - crazy ecf6a3d7ae5a4293bd12171794b09a0f
    - RetroMAE
        - trivial 7b2d20d36b6c41778ac6ac44c13cac66
        - advanced abcf0e71cb7347db8726cf2f119939bf
        - crazy 739a35638b8e4031b747c99954d952f8

## Бенчмарки

Имеются замеренные бенчмарки:
- для моделей BERT, RoBERTA, RetroMAE
- для сетов trivial, advanced, crazy
- для сетов fair, unfair
- для эпох 0 1 2 3 4
- бенчмарки multiclass, multilabel

Замеры для бейзлайнов:
- raw: без контрастивного обучения (CL), просто претрейненные веса
- halves: CL, где положительная пара получена путём деления одного диалога на две части

В будущем:
- бенчмарк downstream classification
- бенчмарк MTEB (?)
- разные виды CL: symmetric, cross, bce (?)
- добавить дропаут над эмбедингом в качестве аугментации (?)

## Сравнение

Эксперимент 1:
- для каждой модели (BERT/RoBERTa/RetroMAE)
- для последней эпохи (или в среднем для всех эпох)
- для fair
- сравнить сеты raw, halves, trivial, advanced, crazy
- на бенчмарках multiclass, multilabel
- оптимистическая гипотеза: raw < halves < trivial < advanced

Эксперимент 2:
- для каждой модели
- для последней эпохи
- выбрать лучший сет среди raw, halves и тд
- сравнить fair и unfair
- на бенчмарках multiclass, multilabel
- оптимистическая гипотеза: fair дает не сильно хуже качество

Эксперимент 3:
- для каждой модели
- для лучшего сета
- сравнить эпохи
- на бенчмарках multiclass, multilabel
- оптимистическая гипотеза: существует оптимальное число эпох без переобучения и со значительным приростом относительно raw и halves

Эксперимент 4:
- для лучшего сета
- для последней эпохи
- для fair
- сравнить все модели до и после CL 
- на бенчмарках multiclass, multilabel
- оптимистическая гипотеза: худшая модель после обучения не хуже чем лучшая модель до обучения
