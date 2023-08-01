import numpy as np
import json
from visualization_utils import read_csv
from sentence_encoding import sentence_encoder
from dgac_clustering import Clusters
import pickle


def count_vectorizer_from_sile_system(names):
    """
    1. Encodes each utterance with sentence encoder specified in sentence_encoding.py
    2. Infers DGAC clusterer to predict labels for each utterance
    3. Saves them to json as f'aug-data/clust-labels-{name}.json'
    4. Vectorizes each dialogue as bag of cluster labels (analogue to bag of words)
    5. Saves vectorizations to .npy files as f'aug-data/vectors-{name}'

    Params
    ------
    - names: list[str], is used to read original and augmented dialogues in format f'aug-data/{name}.csv'
    """
    speaker = np.array(json.load(open('aug-data/speaker.json', 'r')))
    clusterer: Clusters = pickle.load(open('clust-data/dgac_clusterer.pickle', 'rb'))
    rle = json.load(open('aug-data/rle.json', 'r'))

    for name in names:
        utterances = read_csv(f'aug-data/{name}.csv')
        embeddings = sentence_encoder(utterances)
        
        labels_user, labels_system = clusterer.predict(embeddings, speaker)
        labels = np.empty(len(embeddings), dtype=np.int16)
        labels[speaker == 0] = labels_user
        labels[speaker == 1] = labels_system + clusterer.n_clusters // 2
        json.dump(labels.tolist(), open(f'aug-data/clust-labels-{name}.json', 'w'))

        vectors = []
        for i in range(len(rle)):
            start = sum(rle[:i])
            end = start + rle[i]
            vectors.append(np.bincount(labels[start:end], minlength=clusterer.n_clusters))
        
        np.save(f'aug-data/vectors-{name}', vectors)


def count_vectorizer_from_argument(augmented_utterances, rle, speaker, clusterer):
    """
    1. Encodes each utterance with sentence encoder specified in sentence_encoding.py
    2. Infers DGAC clusterer to predict labels for each utterance
    3. Vectorizes each dialogue as bag of cluster labels (analogue to bag of words)

    Params
    ------
    - augmented_utterances: list[str], all utterances merged into single list
    - rle: list[int], list of lengths of dialogues
    - speaker: np.ndarray, list of 0 and 1 corresponding to user and system utterances
    - clusterer: Clusters, pretrained dgac clusterer
    """
    embeddings = sentence_encoder(augmented_utterances)
    
    labels_user, labels_system = clusterer.predict(embeddings, speaker)
    labels = np.empty(len(embeddings), dtype=np.int16)
    labels[speaker == 0] = labels_user
    labels[speaker == 1] = labels_system + clusterer.n_clusters // 2

    vectors = []
    for i in range(len(rle)):
        start = sum(rle[:i])
        end = start + rle[i]
        vectors.append(np.bincount(labels[start:end], minlength=clusterer.n_clusters))
    
    return vectors


def intersection(vec_i, vec_j):
    return np.minimum(vec_i, vec_j).sum()


def dice(vec_i, vec_j):
    return 2 * np.minimum(vec_i, vec_j).sum() / (vec_i.sum() + vec_j.sum())


def compute_similarities_from_file_system(name, func=dice):
    orig_vecs = np.load(f'aug-data/vectors-original.npy')
    aug_vecs = np.load(f'aug-data/vectors-{name}.npy')
    
    return ' '.join([f'{func(orig, aug):.2f}' for orig, aug in zip(orig_vecs, aug_vecs)])


def compute_similarities_from_argument(vec_1, vec_2, func=dice):
    return [func(vec_a, vec_b) for vec_a, vec_b in zip(vec_1, vec_2)]