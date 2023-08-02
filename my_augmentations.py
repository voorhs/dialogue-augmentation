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
from typing import List


def get_dialogues() -> List[List[str]]:
    rle = json.load(open('aug-data/rle.json', 'r'))
    utterances = read_csv('aug-data/original.csv')
    res = []
    for i in range(len(rle)):
        start = sum(rle[:i])
        end = start + rle[i]
        res.append(utterances[start:end])
    return res


class BackTranslator:
    def __init__(self, language, forbidden_tokens=None, device='cpu'):
        self.language = language
        self.device = device

    def from_file_system(self, name='back_trans_hf'):
        """
        Params
        ------
        - name: str, name of output .csv file
        """

        # to french
        translator = pipeline('translation_en_to_fr', model=f'Helsinki-NLP/opus-mt-en-{self.language}', device=self.device)
        original = read_csv('aug-data/original.csv')
        translated = [a['translation_text'] for a in translator(original)]
        del translator

        # back to english
        translator = pipeline('translation_fr_to_en', model=f'Helsinki-NLP/opus-mt-{self.language}-en', device=self.device)
        back_translated = [a['translation_text'] for a in translator(translated)]
        del translator

        # save to csv
        df = pd.DataFrame({'text': back_translated})
        df.to_csv(f'aug-data/{name}.csv')
    
    def from_argument(self, dialogues):
        """
        Params
        ------
        - dialogues: list[list[str]]
        """

        # to french
        translator = pipeline('translation_en_to_fr', model=f'Helsinki-NLP/opus-mt-en-{self.language}', device=self.device)
        original = []
        for dia in dialogues:
            original.extend(dia)
        translated = [a['translation_text'] for a in translator(original)]
        del translator

        # back to english
        translator = pipeline('translation_fr_to_en', model=f'Helsinki-NLP/opus-mt-{self.language}-en', device=self.device)
        back_translated = [a['translation_text'] for a in translator(translated)]

        return back_translated


class Inserter:
    def __init__(
            self,
            fraction=0.1,
            score_threshold=0.005,
            k=5,
            mask_utterance_level=False,
            fill_utterance_level=True,
            model='xlnet-base-cased',
            device='cpu',
            forbidden_tokens=None
        ):
        """
        Params
        ------
        - fraction: float in (0,1), fraction of words by which to increase the length of the dialogues
        - score_thresold: float, lower bound for probability of filled token
        - k: int, parameter for topk sampling
        - mask_utterance_level: bool, whether to mask dialogues as whole or mask each utterance separately
        - fill_utterance_level: bool or int > 1, whether to fill masks in dialogues as whole or process each utterance separately or use context of previous utterances
        - model: str, fill-mask model from hugging face
        - forbidden_tokens: list[str], list of all tokens which won't be used as insertions
        """

        self.fraction = fraction
        self.score_threshold = score_threshold
        self.k = k
        self.mask_utterance_level = mask_utterance_level
        self.fill_utterance_level = fill_utterance_level
        self.model = model
        self.device = device

        if forbidden_tokens is None:
            nltk.download('stopwords')
            forbidden_tokens = stopwords.words('english')
            forbidden_tokens.extend(AutoTokenizer.from_pretrained(self.model).all_special_tokens)
        self.forbidden_tokens = forbidden_tokens

    def _insert(self, words):
        """Insert <mask> after each space"""
        n = len(words)
        size = np.ceil(n * self.fraction).astype(int)
        i_places = np.sort(np.random.choice(n, size=size, replace=False)) + np.arange(size)
        for i in i_places:
            words.insert(i, '<mask>')
        return words

    def _insert_masks_dialogue_level(self, dialogues) -> List[str]:
        """
        Insert <mask> into random places of dialogues

        Params
        ------
        - dialogues: list[list[str]] 

        Return
        ------
        list of dialogues, where each dialogue is a single string with \\n delimiter between utterances
        """

        res = []
        
        for dia in dialogues:
            original = '\n'.join(dia)
            words = self._insert(original.split(' '))
            res.append(' '.join(words))
        
        return res

    def _insert_masks_utterance_level(self, dialogues) -> List[str]:
        """
        Insert <mask> into random places of dialogues

        Params
        ------
        - dialogues: list[list[str]] 

        Return
        ------
        list of dialogues, where each dialogue is a single string with \\n delimiter between utterances
        """

        res = []

        for dia in dialogues:
            masked = []
            for ut in dia:
                words = self._insert(ut.split(' '))
                masked.append(' '.join(words))
                
            res.append('\n'.join(masked))
        
        return res

    def _is_not_forbidden(self, word: str) -> bool:
        word = word.lower()
        flags = []
        flags.append(word in self.forbidden_tokens)
        flags.append(not word.isalpha())
        return not any(flags)

    def _choose_confident(self, fill_res):
        """
        Drop predicted tokens which have low score or are included into forbidden tokens.
        
        Params
        ------
        - fill_res: predicted tokens for single <mask>
        """
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
        return words, scores
    
    def _replace_masks(self, text, outputs) -> str:
        """Replace <mask> with predicted tokens."""
        for words, scores in outputs:
            i = text.find('<mask>')
            if len(words) == 0:
                if text[i-1] == ' ':
                    text = text[:i-1] + text[i+6:]
                else:
                    text = text[:i] + text[i+7:]
            else:
                probs = np.array(scores) / sum(scores)
                to_insert = words[int(np.random.choice(len(words), 1, p=probs))]
                text = text[:i] + to_insert + text[i+6:]
        return text

    def _fill_masks_dialogue_level(self, masked_dialogues) -> List[str]:
        """
        Apply MLM to fill <mask> in given dialogues.

        Params
        ------
        - masked_dialogues: list of dialogues i.e. strings with \\n delimiter betweem utterances
 
        Return
        ------
        list of utterances merged into single list
        """

        mask_filler = pipeline('fill-mask', model=self.model, device=self.device)
        dataset_fill_results = mask_filler(masked_dialogues, top_k=1000)
        res = []
        for dia, dia_fill_results in zip(masked_dialogues, dataset_fill_results):
            # choose only confident predictions
            outputs = [self._choose_confident(mask_fill_results) for mask_fill_results in dia_fill_results]
                
            # insert predictions
            res.extend(self._replace_masks(dia, outputs).split('\n'))
        
        return res

    def _fill_masks_utterance_level(self, masked_dialogues) -> List[str]:
        """
        Apply MLM to fill <mask> in given dialogues.

        Params
        ------
        - masked_dialogues: list of dialogues, where each dialogue is a single string with \\n delimiter between utterances

        Return
        ------
        list of utterances merged into single list
        """

        # get single list of utterances
        utterances = []
        for dia in masked_dialogues:
            utterances.extend(dia.split('\n'))
        
        # separate those with and without <mask>
        i_uts_without_mask = []
        uts_with_mask = []
        for i, ut in enumerate(utterances):
            if ut.find('<mask>') == -1:
                i_uts_without_mask.append(i)
            else:
                uts_with_mask.append(ut)

        # feed to MLM
        mask_filler = pipeline('fill-mask', model=self.model, device=self.device)
        dataset_fill_results = mask_filler(uts_with_mask, top_k=1000)
        
        # insert predictions to utterances with <mask>
        res = []
        for ut, ut_fill_results in zip(uts_with_mask, dataset_fill_results):
            if isinstance(ut_fill_results[0], dict):
                ut_fill_results = [ut_fill_results]
            candidates = [self._choose_confident(mask_fill_results) for mask_fill_results in ut_fill_results]
            res.append(self._replace_masks(ut, candidates))

        # merge masked and untouched utterances
        for i in i_uts_without_mask:
            res.insert(i, utterances[i])

        return res
    
    def _fill_masks_context_level(self, masked_dialogues, context_length) -> List[str]:
        """
        Apply MLM to fill <mask> in given dialogues.

        Params
        ------
        - masked_dialogues: list of dialogues, where each dialogue is a single string with \\n delimiter between utterances

        Return
        ------
        list of utterances merged into single list
        """

        # get single list of utterances
        context_list = []
        for dia in masked_dialogues:
            dia = dia.split('\n')
        
            # join consequetive utterances into single string
            context_list.extend(['\n'.join(dia[i:i+context_length]) for i in range(0, len(dia), context_length)])

        # feed to MLM
        mask_filler = pipeline('fill-mask', model=self.model, device=self.device)
        dataset_fill_results = mask_filler(context_list, top_k=1000)
        
        # insert predictions to utterances with <mask>
        res = []
        for context, context_fill_results in zip(context_list, dataset_fill_results):
            if isinstance(context_fill_results[0], dict):
                context_fill_results = [context_fill_results]
            candidates = [self._choose_confident(mask_fill_results) for mask_fill_results in context_fill_results]
            res.extend(self._replace_masks(context, candidates).split('\n'))

        return res

    def from_file_system(self, name):
        """
        Add words to random places of dialogues.
        
        Reads data from from 'aug-data/original.csv'. Saves result to f'aug-data/{name}.csv'.
        """

        # load data
        dialogues = get_dialogues()

        # perform masking and insertion
        if self.mask_utterance_level:
            masked = self._insert_masks_utterance_level(dialogues)
        else:
            masked = self._insert_masks_dialogue_level(dialogues)
        
        if not isinstance(self.fill_utterance_level, bool):
            filled = self._fill_masks_context_level(masked, self.fill_utterance_level)
        elif self.fill_utterance_level:
            filled = self._fill_masks_utterance_level(masked)
        else:
            filled = self._fill_masks_dialogue_level(masked)
        
        # save result
        df = pd.DataFrame({'text': filled})
        df.to_csv(f'aug-data/{name}.csv')
    
    def from_argument(self, dialogues):
        """
        Add words to random places of dialogues from 'aug-data/original.csv'.

        Params
        ------
        - dialogues: list[list[str]]

        Return
        ------
        list of utterances merged into single list
        """

        if self.mask_utterance_level:
            masked = self._insert_masks_utterance_level(dialogues)
        else:
            masked = self._insert_masks_dialogue_level(dialogues)
        
        if isinstance(self.fill_utterance_level, int):
            filled = self._fill_masks_context_level(masked, self.fill_utterance_level)
        elif self.fill_utterance_level:
            filled = self._fill_masks_utterance_level(masked)
        else:
            filled = self._fill_masks_dialogue_level(masked)
        
        return filled


class Replacer(Inserter):
    def __init__(
            self,
            k=3,
            fill_utterance_level=True,
            model='xlnet-base-cased',
            device='cpu',
            forbidden_tokens=None
        ):
        super().__init__(
            fraction=1,
            score_threshold=0,
            k=k,
            mask_utterance_level=False,
            fill_utterance_level=fill_utterance_level,
            model=model,
            device=device,
            forbidden_tokens=forbidden_tokens
        )
        self.replaced_tokens = []

    def _insert(self, words):
        for i, word in enumerate(words):
            if self._is_not_forbidden(word):
                self.replaced_tokens.append(word)
                words[i] = '<mask>'
        return words
    
    def _replace_masks(self, text, outputs):
        for words, scores in outputs:
            i = text.find('<mask>')
            to_insert = self.replaced_tokens.pop(0)
            if len(words) > 0:
                probs = np.array(scores) / sum(scores)
                to_insert = words[int(np.random.choice(len(words), 1, p=probs))]
            text = text[:i] + to_insert + text[i+6:]
        return text
