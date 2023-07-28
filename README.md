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
- FoCus
- PersonaChat
- DailyDialogue