from transformers import pipeline
from visualization_utils import read_csv
import pandas as pd
from visualization_utils import get_dialogue
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import string


def back_translate(name='back_trans_hf'):
    """
    Params
    ------
    - name: str, name of output .csv file
    """

    # to french
    translator = pipeline('translation_en_to_fr', device='cuda:0')
    original = read_csv('aug-data/original.csv')
    translated = [a['translation_text'] for a in translator(original)]
    del translator

    # back to english
    translator = pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en', device='cuda:0')
    back_translated = [a['translation_text'] for a in translator(translated)]
    del translator

    # save to csv
    df = pd.DataFrame({'text': back_translated})
    df.to_csv(f'aug-data/{name}.csv')


class Inserter:
    def __init__(self, fraction=0.1, score_threshold=0.005, k=5, utterance_level=True):
        """
        Params
        ------
        - fraction: float in (0,1), fraction of words by which to increase the length of the dialogues
        - score_thresold: float, lower bound for probability of filled token
        - k: int, parameter for topk sampling
        - utterance_level: bool, whether to mask dialogues as whole or mask each utterance separately
        """

        self.fraction = fraction
        self.score_threshold = score_threshold
        self.k = k
        self.utterance_level = utterance_level

        nltk.download('stopwords')
        self.stopwords_list = stopwords.words('english')
        self.special = AutoTokenizer.from_pretrained('xlnet-base-cased').all_special_tokens

    def _insert_masks_dialogue_level(self) -> list[str]:
        """
        Insert <mask> into random places of dialogues from 'aug-data/original.csv'

        Return
        ------
        list of dialogues, where each dialogue is a single string with \\n delimiter between utterances
        """
        res = []
        n_dialogues = len(json.load(open('aug-data/rle.json', 'r')))
        
        for i in range(n_dialogues):
            # concat utterances into single string
            original = '\n'.join(get_dialogue(i, 'original'))

            # insert masked token after each space
            words = original.split(' ')
            n = len(words)
            size = np.ceil(n * self.fraction).astype(int)
            i_places = np.random.choice(n, size=size) + np.arange(size)
            for i in i_places:
                words.insert(i, '<mask>')
            
            # concat words
            res.append(' '.join(words))
        
        return res

    def _insert_masks_utterance_level(self) -> list[list[str]]:
        """
        Insert <mask> into random places of dialogues from 'aug-data/original.csv'

        Return
        ------
        list of dialogues, where each dialogue is list of strings (utterances)
        """

        res = []
        n_dialogues = len(json.load(open('aug-data/rle.json', 'r')))

        for i in range(n_dialogues):
            original = get_dialogue(i, 'original')
            masked = []
            for ut in original:
                # insert <mask> after each space
                words = ut.split(' ')
                n = len(words)
                size = np.ceil(n * self.fraction).astype(int)
                i_places = np.random.choice(n, size=size) + np.arange(size)
                for i in i_places:
                    words.insert(i, '<mask>')

                # concat words
                masked.append(' '.join(words))
                
            res.append(masked)
        
        return res

    def _is_not_forbidden(self, word) -> bool:
        word = word.lower()
        flags = []
        flags.append(word in self.stopwords_list)
        flags.append(word in self.special)
        flags.append(word in string.whitespace)
        flags.append(word in string.punctuation)
        return not any(flags)

    def _fill_masks_dialogue_level(self, masked_dialogues) -> list[str]:
        """
        Apply MLM to fill <mask> in given dialogues.

        Params
        ------
        - masked_dialogues: list of utterances
        """

        mask_filler = pipeline('fill-mask', model='xlnet-base-cased')
        
        res = []
        for dia in masked_dialogues:
            # choose only confident predictions
            outputs = []
            for fill_res in mask_filler(dia, top_k=1000):
                words = []
                scores = []
                for word, score in map(lambda x: (x['token_str'], x['score']), fill_res):
                    if len(words) == self.k:
                        break
                    if score < self.score_threshold:
                        continue
                    if self._is_not_forbidden(word):
                        words.append(word)
                        scores.append(score)
                outputs.append((words, scores))
            # insert predictions
            for words, scores in outputs:
                i = dia.find('<mask>')
                if len(words) == 0:
                    if dia[i-1] == ' ':
                        dia = dia[:i-1] + dia[i+6:]
                    else:
                        dia = dia[:i] + dia[i+7:]
                else:
                    probs = np.array(scores) / sum(scores)
                    to_insert = words[int(np.random.choice(len(words), 1, p=probs))]
                    dia = dia[:i] + to_insert + dia[i+6:]
            
            res.extend(dia.split('\n'))
        
        return res

    def _fill_masks_utterance_level(self, masked_dialogues) -> list[str]:
        """
        Apply MLM to fill <mask> in given dialogues.

        Params
        ------
        - masked_dialogues: list of dialogues, where each dialogue is list of strings (utterances)
        """

        mask_filler = pipeline('fill-mask', model='xlnet-base-cased')
        
        res = []
        for dia in masked_dialogues:
            # choose only confident predictions
            outputs = []
            for ut in dia:
                all_masks = mask_filler(ut, top_k=1000)
                if isinstance(all_masks[0], dict):
                    # in case there's single mask in utterance
                    all_masks = [all_masks]
                options = []
                for fill_res in all_masks:
                    words = []
                    scores = []
                    for word, score in map(lambda x: (x['token_str'], x['score']), fill_res):
                        if len(words) == self.k:
                            break
                        if score < self.score_threshold:
                            continue
                        if self._is_not_forbidden(word):
                            words.append(word)
                            scores.append(score)
                    options.append((words, scores))
                outputs.append(options)
            
            # insert predictions
            filled = []
            for ut, options in zip(dia, outputs):
                for words, scores in options:
                    i = ut.find('<mask>')
                    if len(words) == 0:
                        if ut[i-1] == ' ' and i != 0:
                            ut = ut[:i-1] + ut[i+6:]
                        else:
                            ut = ut[:i] + ut[i+7:]
                    else:
                        probs = np.array(scores) / sum(scores)
                        to_insert = words[int(np.random.choice(len(words), 1, p=probs))]
                        ut = ut[:i] + to_insert + ut[i+6:]
                filled.append(ut)
            
            res.extend(filled)

        return res

    def __call__(self, name):
        """
        Add words to random places of dialogues from 'aug-data/original.csv'.

        Params:
        - name: str, name of output .csv file
        """
        if self.utterance_level:
            res = self._fill_masks_utterance_level(self._insert_masks_utterance_level())
        else:
            res = self._fill_masks_dialogue_level(self._insert_masks_dialogue_level())
        

        df = pd.DataFrame({'text': res})
        df.to_csv(f'aug-data/{name}.csv')
