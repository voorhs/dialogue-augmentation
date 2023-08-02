import optuna
from my_augmentations import Inserter
from dgac_clustering import Clusters
import numpy as np
import json
import pickle
from my_augmentations import Inserter, Replacer, BackTranslator, get_dialogues
from similarity_functions import count_vectorizer_from_argument as cv, compute_similarities_from_argument as cs
from time import time
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from Levenshtein import distance as edit_distance, ratio as normalized_edit_similarity, jaro as jaro_similarity


speaker = np.array(json.load(open('aug-data/speaker.json', 'r')))
clusterer: Clusters = pickle.load(open('clust-data/dgac_clusterer.pickle', 'rb'))
rle = json.load(open('aug-data/rle.json', 'r'))
vectors_original = np.load('aug-data/vectors-original.npy')

nltk.download('stopwords')
forbidden_tokens = stopwords.words('english')
forbidden_tokens.extend(AutoTokenizer.from_pretrained('microsoft/mpnet-base').all_special_tokens)


inserter_search_space = {
    'fraction': [0.1, 0.3, 0.5, 0.7],
    'score_threshold': [0, 0.0025, 0.005, 0.01, 0.013],
    'k': [1, 3, 5, 10],
    'mask_utterance_level': [False, True],
    'fill_utterance_level': [True, 2, 3, 4],
    'model': ['microsoft/mpnet-base']
}

replacer_search_space = {
    'k': [1, 3, 5, 10],
    'mask_utterance_level': [False],
    'fill_utterance_level': [True, 2, 3, 4],
    'model': ['microsoft/mpnet-base']
}

back_translator_search_space = {
    'language': [
        'am', 'ar', 'eu', 'bn', 'bg', 'ca', 'hr', 'cs', 'da', 'nl',
        'et', 'fi', 'fr', 'de', 'el', 'gu', 'iw', 'hi', 'hu',
        'is', 'id', 'it', 'ja', 'kn', 'ko', 'lv', 'lt', 'ms', 'ml',
        'mr', 'no', 'pl', 'ro', 'ru', 'sr', 'sk', 'sl', 'es', 'sw',
        'sv', 'ta', 'te', 'th', 'tr', 'ur', 'uk', 'vi', 'cy']
}


def statistics(trial: optuna.trial.Trial, search_space, Augmenter, dialogues):
    hyperparams = {key: trial.suggest_categorical(key, val) for key, val in search_space.items()}
    begin = time()
    inserter = Augmenter(
        forbidden_tokens=forbidden_tokens,
        device='cuda',
        **hyperparams
    )
    augmented = inserter.from_argument(dialogues)
    total_time = time() - begin
    vectors_augmented = cv(augmented, rle, speaker, clusterer)
    similarities = cs(vectors_augmented, vectors_original)

    mean = np.mean(similarities)
    med = np.median(similarities)
    var = np.var(similarities)

    dials = []
    for dia in dialogues:
        dials.extend(dia)
    orig = ' '.join(dials)
    aug = ' '.join(augmented)
    dist = edit_distance(orig, aug)
    sim1 = normalized_edit_similarity(orig, aug)
    sim2 = jaro_similarity(orig, aug)

    return mean, med, var, total_time, dist, sim1, sim2


if __name__ == "__main__":
    from optuna.samplers import GridSampler
    import logging
    import sys
    import argparse
    from functools import partial


    ap = argparse.ArgumentParser()
    ap.add_argument('--augmenter', dest='augmenter', required=True, choices=['inserter', 'replacer', 'back_translator'])
    ap.add_argument('--study_name', dest='study_name', required=True)
    args = ap.parse_args()


    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = f"sqlite:///{args.study_name}.db"

    if args.augmenter == 'inserter':
        search_space = inserter_search_space
        Augmenter = Inserter
    elif args.augmenter == 'replacer':
        search_space = replacer_search_space
        Augmenter = Replacer
    elif args.augmenter == 'back_translator':
        search_space = back_translator_search_space
        Augmenter = BackTranslator
    
    study = optuna.create_study(
        sampler=GridSampler(search_space, seed=0),
        directions=['maximize']*7,
        study_name=args.study_name,
        storage=storage_name,
        load_if_exists=True
    )
    
    # n_trials is equal to number of all possible hparams combinations
    n_trials = 1
    for val in search_space.values():
        if isinstance(val, list):
            n_trials *= len(val)

    dialogues = get_dialogues()
    func = partial(statistics, search_space=search_space, Augmenter=Augmenter, dialogues=dialogues)
    
    study.optimize(func, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
