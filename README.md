# Dialogue Augmentation

Структура репы:
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

- кажется идеальным вариантом:
    - взять байден (или другую модель для dialogue modeling), зафайнтюнить его таким образом, чтобы на уровне диалога аттеншен был только между cls токенами каждого уттеранса, а внутри каждого уттеранса обычный аттеншен
    - закодить такую модель будет небыстро, поэтому сделать это стоит только после тестов над текущим вариантом уттеранс ранкера
    - можно и без байдена!