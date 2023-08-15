from visualization_utils import read_csv
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import string
from typing import List, Literal


def get_dialogues(return_speaker=False) -> List[List[str]]:
    rle = json.load(open('aug-data/rle.json', 'r'))
    utterances = read_csv('aug-data/original.csv')
    res = []
    if return_speaker:
        speaker = json.load(open('aug-data/speaker.json', 'r'))
        res_speakers = []
    for i in range(len(rle)):
        start = sum(rle[:i])
        end = start + rle[i]
        res.append(utterances[start:end])
        if return_speaker:
            res_speakers.append(speaker[start:end])
    if return_speaker:
        return res, res_speakers
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

        # separate those with and without <mask>
        i_cont_without_mask = []
        cont_with_mask = []
        for i, ut in enumerate(context_list):
            if ut.find('<mask>') == -1:
                i_cont_without_mask.append(i)
            else:
                cont_with_mask.append(ut)

        # feed to MLM
        mask_filler = pipeline('fill-mask', model=self.model, device=self.device)
        dataset_fill_results = mask_filler(cont_with_mask, top_k=1000)
        
        # insert predictions to contexts with <mask>
        res = []
        for context, context_fill_results in zip(cont_with_mask, dataset_fill_results):
            if isinstance(context_fill_results[0], dict):
                context_fill_results = [context_fill_results]
            candidates = [self._choose_confident(mask_fill_results) for mask_fill_results in context_fill_results]
            res.append(self._replace_masks(context, candidates))

        # merge masked and untouched utterances
        for i in i_cont_without_mask:
            res.insert(i, context_list[i])
        
        # roll out contexts
        res_uts = []
        for context in res:
            res_uts.extend(context.split('\n'))

        return res_uts

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


def _post_process(sequences, prompt_list, as_json):
    """Helper for Llama augmenters."""
    res = []
    for seq, prompt in zip(sequences, prompt_list):
        txt = seq['generated_text'][len(prompt):]
        if as_json:
            start = txt.find('{')
            end = txt.rfind('}')+1
            cur_res = json.loads(txt[start:end])
        else:
            start = txt.find('"""')
            end = txt.rfind('```')+1
            cur_res = txt[start:end]
        res.append(cur_res)
    return res


class LlamaMaskFiller:
    def __init__(self, llm, tokenizer, masking: Literal['replace', 'insert', 'head', 'tail'], fraction=0.1, as_json=True):
        self.masking = masking
        self.fraction = fraction
        self.as_json = as_json

        self.llm = llm
        self.tokenizer = tokenizer

        print(self)

    def _prompt(self, dialogue, speaker):
        """
        Make prompt for mask filling task.

        Params
        ------
            dialogue: list[str], list of dialogue utterances
            speaker: list[int], list of IDs of speaker for each utterance
            as_json: whether to give result as a json (works very poorly)
        """

        if self.as_json:
            utterances = json.dumps({'utterances': dialogue, 'speaker': speaker})
            system = """You work as a function for dialogue completion. Input is a json dictionaty with keys 'utterances' and 'speaker'. Former item is a list of dialogue utterances, latter item is a list of IDs of speakers corresponding to each utterance.

You need to replace all [[LOST]] tokens in utterances with generated meaningful utterances that match the context of the dialogue. Firstly, construct json list `utterances`, then look through it and construct `speaker` to ensure that it contains all speakers IDs corresponding to resulting utterances in a correct order and count. Ensure that json format is correct.

You must not give any comments, greetings or explanations, only resulting json following the format of input."""

        else:
            utterances = '\n'.join([f'[{"AB"[i]}] ```{ut}```' for i, ut in zip(speaker, dialogue)])
            system = """You work as a function with specific input text and desired output text. Do not give any comments, greetings or explanations, only desired output.

Specific input is a dialogue prodived by a user. Each turn of the dialogue begins with [A] or [B] denoting the speaker, followed by the utterance in triple quotes. Some utterances are lost, this is indicated by the [[LOST]] token.

You need to replace all [[LOST]] tokens with generated meaningful utterances that match the context of the dialogue.

Desired output is the utterances generated in the places where [[LOST]] tokens were. It means you must not rewrite entire dialogue."""

        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

Specific input:
\"\"\"
{utterances}
\"\"\"
[/INST]"""

    @staticmethod
    def _replace(dialogue, speaker, fraction):
        dialogue = dialogue[:]
        n = len(dialogue)
        size = np.ceil(n * fraction).astype(int)
        i_places = np.random.choice(n, size=size, replace=False)
        for i in range(n):
            if i in i_places:
                dialogue[i] = '[[LOST]]'
            else:
                dialogue[i] = f'{dialogue[i]}'
        return dialogue, speaker

    @staticmethod
    def _tail(dialogue: List[str], speaker, fraction):
        dialogue = dialogue[:]
        speaker = speaker[:]
        n = len(dialogue)
        size = np.ceil(n * fraction).astype(int)
        role = speaker[-1]
        for i in range(n, n+size):
            dialogue.insert(i, '[[LOST]]')
            role = 1-role
            speaker.insert(i, role)
            
        return dialogue, speaker

    @staticmethod
    def _insert(dialogue: List[str], speaker, fraction):
        dialogue = dialogue[:]
        n = len(dialogue)
        size = np.ceil(n * fraction).astype(int)
        i_places = np.sort(np.random.choice(n, size=size, replace=False)) + np.arange(size) * 2
        for i in i_places:
            dialogue.insert(i, '[[LOST]]')
            dialogue.insert(i, '[[LOST]]')
            role = speaker[i-1]
            speaker.insert(i, role)
            speaker.insert(i, 1-role)
        return dialogue, speaker
    
    @staticmethod
    def _head(dialogue: List[str], speaker, fraction=0.2):
        dialogue = dialogue[:]
        speaker = speaker[:]
        n = len(dialogue)
        size = np.ceil(n * fraction).astype(int)
        role = speaker[0]
        for i in range(size):
            dialogue.insert(i, '[[LOST]]')
            role = 1-role
            speaker.insert(i, role)
            
        return dialogue, speaker

    def from_file_system(self, name):
        dialogues, speakers = get_dialogues(return_speaker=True)

        if self.masking == 'replace':
            masker = LlamaMaskFiller._replace
        elif self.masking == 'insert':
            masker = LlamaMaskFiller._insert
        elif self.masking == 'head':
            masker = LlamaMaskFiller._head
        elif self.masking == 'tail':
            masker = LlamaMaskFiller._tail
        else:
            raise ValueError(f'Unknown masking: {self.masking}')

        prompt_list = []
        for dia, spe in zip(dialogues, speakers):
            dia, spe = masker(dia, spe, self.fraction)
            prompt_list.append(self._prompt(dia, spe))

        sequences = self.llm(
            prompt_list,
            num_return_sequences=1,
            max_new_tokens=1024,
            batch_size=2,
            
            do_sample=True,
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=1
        )

        json.dump(sequences, open(f'aug-data/{name}-raw.json', 'w'))
        
        try:
            processed = _post_process(sequences, prompt_list, self.as_json)
            json.dump(processed, open(f'aug-data/{name}.json', 'w'))
        except Exception as e:
            print(f'error occurred during post processing: {e}')


class LlamaSummarizer:
    def __init__(self, penalty_length, llm, tokenizer, as_json=True):
        self.penalty_length = penalty_length
        self.as_json = as_json
        self.llm = llm
        self.tokenizer= tokenizer

        print(self)
    
    def _prompt(self, dialogue, speaker):
        """
        Make prompt for mask filling task.

        Params
        ------
            dialogue: list[str], list of dialogue utterances
            speaker: list[int], list of IDs of speaker for each utterance
            as_json: whether to give result as a json (works very poorly)
        """

        if self.as_json:
            utterances = json.dumps({'utterances': dialogue, 'speaker': speaker})
            system = """You work as a function for dialogue completion. Input is a json dictionaty with keys 'utterances' and 'speaker'. Former item is a list of dialogue utterances, latter item is a list of IDs of speakers corresponding to each utterance.

You need to replace all [[LOST]] tokens in utterances with generated meaningful utterances that match the context of the dialogue. Firstly, construct json list `utterances`, then look through it and construct `speaker` to ensure that it contains all speakers IDs corresponding to resulting utterances in a correct order and count. Ensure that json format is correct.

You must not give any comments, greetings or explanations, only resulting json following the format of input."""

        else:
            utterances = '\n'.join([f'[{"AB"[i]}] ```{ut}```' for i, ut in zip(speaker, dialogue)])
            system = f"""You work as a function for dialogue text transformation with specific input text and desired output text.

Specific input is a dialogue prodived by a user. Each turn of the dialogue begins with [A] or [B] denoting the speaker, followed by the utterance in triple quotes.

You need to summarize the dialogue by making new one. In total from all speakers, new dialogue maximum number of turns must be two times less than original number of turns. For example, if there are 12 turns in the original dialogue, then there are no more than 6 turns in the new dialogue. New dialogue must preserve meaningfulness and general context of original dialogue.

Desired output is a resulting dialogue following the input format of the original dialogue."""

        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

Specific input:
\"\"\"
{utterances}
\"\"\"
[/INST]"""

    def from_file_system(self, name):
        dialogues, speakers = get_dialogues(return_speaker=True)

        prompt_list = []
        for dia, spe in zip(dialogues, speakers):
            prompt_list.append(self._prompt(dia, spe))
    
        sequences = self.llm(
            prompt_list,
            num_return_sequences=1,
            max_new_tokens=1024,
            batch_size=2,
            
            do_sample=True,
            top_k=50,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.4,
            length_penalty=self.penalty_length
        )

        json.dump(sequences, open(f'aug-data/{name}-raw.json', 'w'))
        
        try:
            processed = _post_process(sequences, prompt_list, self.as_json)
            json.dump(processed, open(f'aug-data/{name}.json', 'w'))
        except Exception as e:
            print(f'error occurred during post processing: {e}')


class LlamaVerbose(LlamaSummarizer):
    def _prompt(self, dialogue, speaker):
        """
        Make prompt for mask filling task.

        Params
        ------
            dialogue: list[str], list of dialogue utterances
            speaker: list[int], list of IDs of speaker for each utterance
            as_json: whether to give result as a json (works very poorly)
        """

        if self.as_json:
            utterances = json.dumps({'utterances': dialogue, 'speaker': speaker})
            system = """You work as a function for dialogue completion. Input is a json dictionaty with keys 'utterances' and 'speaker'. Former item is a list of dialogue utterances, latter item is a list of IDs of speakers corresponding to each utterance.

You need to expand the dialogue by making new one. In total from all speakers, new dialogue minimum number of turns must be 1.5 times more than original number of turns. For example, if there are 12 turns in the original dialogue, then there are at least 18 turns in the new dialogue. New dialogue must preserve meaningfulness and general context of original dialogue.
Firstly, construct json list `utterances`, then look through it and construct `speaker` to ensure that it contains all speakers IDs corresponding to resulting utterances in a correct order and count. Ensure that json format is correct.

You must not give any comments, greetings or explanations, only resulting json following the format of input."""

        else:
            utterances = '\n'.join([f'[{"AB"[i]}] ```{ut}```' for i, ut in zip(speaker, dialogue)])
            system = f"""You work as a function for dialogue text transformation with specific input text and desired output text. Do not give any comments or explanation, only resulting desired output text.

Specific input is a dialogue prodived by a user. Each turn of the dialogue begins with [A] or [B] denoting the speaker, followed by the utterance in triple quotes.

You need to expand the dialogue by making new one. In total from all speakers, new dialogue minimum number of turns must be two times more than original number of turns. For example, if there are 6 turns in the original dialogue, then there are at least 12 turns in the new dialogue. New dialogue must preserve meaningfulness and general context of original dialogue.

Desired output is a resulting dialogue following the input format of the original dialogue."""

        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

Specific input:
\"\"\"
{utterances}
\"\"\"
[/INST]"""


class LlamaParaphraser(LlamaSummarizer):
    def __init__(self, style: Literal['formal', 'informal', 'technical', 'persuasive', 'creative', 'poetic', 'playful'], llm, tokenizer, as_json=True):
        self.style = style
        super().__init__(0, llm, tokenizer, as_json)

    def _prompt(self, dialogue, speaker, as_json=False):
        """
        Make prompt for mask filling task.

        Params
        ------
            dialogue: list[str], list of dialogue utterances
            speaker: list[int], list of IDs of speaker for each utterance
            as_json: whether to give result as a json (works very poorly)
        """

        style_descr = {
            'formal': 'formal style. This type of text is characterized by a more serious and professional tone, often used in formal letters, business proposals, and academic papers.',
            'informal': 'informal style. This type of text is characterized by a more casual and relaxed tone, often used in everyday conversations, social media, and text messages.',
            'technical': 'technical style. This type of text is characterized by the use of technical terms and jargon, often used in instruction manuals, technical reports, and scientific papers.',
            'persuasive': 'persuasive style. This type of text is characterized by the use of rhetorical devices and persuasive techniques, often used in sales and marketing materials, persuasive essays, and opinion pieces.',
            'creative': 'creative style. This type of text is characterized by imaginative and expressive language, often used in poetry, fiction, and creative nonfiction',
            'poetic': 'poetic style. This type of text is characterized by imaginative language and creative expression, often used in poetry, song lyrics, and spoken word performances.',
            'playful': "playful style. This style of dialogue involves humor, wit, and lighthearted teasing. It's characterized by a relaxed and joyful atmosphere, and a willingness to have fun and enjoy each other's company.",
        }

        if as_json:
            utterances = json.dumps({'utterances': dialogue, 'speaker': speaker})
            system = f"""You work as a function for dialogue completion. Input is a json dictionaty with keys 'utterances' and 'speaker'. Former item is a list of dialogue utterances, latter item is a list of IDs of speakers corresponding to each utterance.

You need to construct new dialogue by paraphrasing original dialogue to {style_descr[self.style]}. New dialogue must preserve meaningfulness and general context of original dialogue.
Firstly, construct json list `utterances`, then look through it and construct `speaker` to ensure that it contains all speakers IDs corresponding to resulting utterances in a correct order and count. Ensure that json format is correct.

You must not give any comments, greetings or explanations, only resulting json following the format of input."""

        else:
            utterances = '\n'.join([f'[{"AB"[i]}] ```{ut}```' for i, ut in zip(speaker, dialogue)])
            system = f"""You work as a function for dialogue text transformation with specific input text and desired output text. Do not give any comments or explanation, only resulting desired output text.

Specific input is a dialogue prodived by a user. Each turn of the dialogue begins with [A] or [B] denoting the speaker, followed by the utterance in triple quotes.

You need to construct new dialogue by paraphrasing original dialogue to {style_descr[self.style]}. New dialogue must preserve meaningfulness and general context of original dialogue.

Desired output is a resulting dialogue following the input format of the original dialogue."""

        return f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

Specific input:
\"\"\"
{utterances}
\"\"\"
[/INST]"""


if __name__ == "__main__":

    # inserter = Inserter(
    #     fraction=0.5,
    #     score_threshold=0.005,
    #     k=5,
    #     mask_utterance_level=True,
    #     fill_utterance_level=2,
    #     model='microsoft/mpnet-base',
    #     device='cuda'
    # )
    # inserter.from_file_system('inserter')
    
    # replacer = Replacer(
    #     k=3,
    #     fill_utterance_level=2,
    #     model='microsoft/mpnet-base',
    #     device='cuda'
    # )
    # replacer.from_file_system('replacer')

    # back_translator = BackTranslator(
    #     language='ru',
    #     device='cuda'
    # )
    # back_translator.from_file_system('back_translator')

    model = 'meta-llama/Llama-2-13b-chat-hf'

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(
            model,
            device_map='auto',
            load_in_4bit=True
        ),
        tokenizer=tokenizer
    )

    LlamaMaskFiller(llm, tokenizer, 'replace', fraction=0.2).from_file_system('llm_replacer')
    LlamaMaskFiller(llm, tokenizer, 'insert', fraction=0.2).from_file_system('llm_inserter')
    LlamaMaskFiller(llm, tokenizer, 'head', fraction=0.2).from_file_system('llm_head')
    LlamaMaskFiller(llm, tokenizer, 'tail', fraction=0.2).from_file_system('llm_tail')

    LlamaSummarizer(-5, llm, tokenizer).from_file_system('llm_summarizer')
    LlamaVerbose(+5, llm, tokenizer).from_file_system('llm_verbose')
    LlamaParaphraser('formal', llm, tokenizer).from_file_system('llm_formal')
    LlamaParaphraser('informal', llm, tokenizer).from_file_system('llm_informal')
    LlamaParaphraser('technical', llm, tokenizer).from_file_system('llm_technical')
    LlamaParaphraser('persuasive', llm, tokenizer).from_file_system('llm_persuasive')
    LlamaParaphraser('creative', llm, tokenizer).from_file_system('llm_creative')
    LlamaParaphraser('playful', llm, tokenizer).from_file_system('llm_playful')
    