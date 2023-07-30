# Dialogue Augmentation

## Примеры аугментаций

- вставка stop utterances -- LLM, корпус пар
- переписывание (utters / dialog) - стиль -- LLM (попробовать ChatGPT API в гугл колабе)
- переписывание (utters / dialog) - длина -- LLM
- переписывание (utters / dialog) - по другому -- LLM
- **синонимы** (отдельные слова или фразы) -- MLM
- **перевод** -- MT
- utter -> emb -> utter' -- MPNet
- dialog -> emb -> utter -- ConveRT
- ? utter-> intent -> utter -- классификатор
- ? utter -> emb -> img -> img -> emb -> utter

## Datasets

- MultiWOZ
- AugESC ([ссыла на hf](https://huggingface.co/datasets/thu-coai/augesc))

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