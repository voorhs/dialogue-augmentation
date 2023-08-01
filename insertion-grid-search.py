import optuna
from my_augmentations import Inserter
from dgac_clustering import Clusters
import numpy as np
import json
import pickle
from my_augmentations import Inserter
from similarity_functions import count_vectorizer_from_argument as cv, compute_similarities_from_argument as cs
from time import time
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer


search_space = {
    'fraction': [0.1, 0.2, 0.3, 0.4, 0.5],
    'score_threshold': [0, 0.002, 0.005, 0.01, 0.013],
    'k': [1, 3, 5],
    'mask_utterance_level': [False, True],
    'fill_utterance_level': [False, True],
    'model': ['xlnet-base-cased', 'xlnet-large-cased']
}


speaker = np.array(json.load(open('aug-data/speaker.json', 'r')))
clusterer: Clusters = pickle.load(open('clust-data/dgac_clusterer.pickle', 'rb'))
rle = json.load(open('aug-data/rle.json', 'r'))
vectors_original = np.load('aug-data/vectors-original.npy')

nltk.download('stopwords')
forbidden_tokens = stopwords.words('english')
for model in search_space['model']:
    forbidden_tokens.extend(AutoTokenizer.from_pretrained(model).all_special_tokens)


def statistics(trial: optuna.trial.Trial):
    hyperparams = {key: trial.suggest_categorical(key, val) for key, val in search_space.items()}
    begin = time()
    inserter = Inserter(
        forbidden_tokens=forbidden_tokens,
        device='cuda',
        **hyperparams
    )
    dialogues = inserter._get_dialogues()
    augmented = inserter.from_argument(dialogues)
    total_time = time() - begin
    vectors_augmented = cv(augmented, rle, speaker, clusterer)
    similarities = cs(vectors_augmented, vectors_original)

    mean = np.mean(similarities)
    med = np.median(similarities)
    var = np.var(similarities)
    
    return mean, med, var, total_time


from optuna.samplers import GridSampler
import logging
import sys

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "insertion-grid-search"  # Unique identifier of the study.
storage_name = f"sqlite:///{study_name}.db"

def start_gridsearch(n_jobs, seed=0):
    study = optuna.create_study(
        sampler=GridSampler(search_space, seed=seed),
        directions=['maximize', 'maximize', 'maximize', 'minimize'],
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )
    # n_trials is equal to number of all possible hparams combinations
    n_trials = 1
    for val in search_space.values():
        if isinstance(val, list):
            n_trials *= len(val)
    study.optimize(statistics, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

if __name__ == "__main__":
    start_gridsearch(n_jobs=1)