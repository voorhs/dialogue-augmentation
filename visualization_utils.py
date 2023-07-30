import plotly.express as px
from sklearn.manifold import TSNE
import pickle
from dgac_clustering import Clusters
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from redlines import Redlines
from IPython.display import Markdown, display


def assign_cluster_names(labels, texts) -> list[str]:
    n_clusters = len(np.unique(labels))    

    cluster_utterances = []
    for i in range(n_clusters):
        cluster_texts = [txt for j, txt in enumerate(texts) if labels[j] == i]
        cluster_utterances.append('. '.join(cluster_texts))

    vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    vec = vectorizer.fit_transform(cluster_utterances).toarray()

    titles = np.zeros_like(vec)
    for i, topk in enumerate(np.argpartition(vec, kth=[-1, -2, -3], axis=1)[:, -3:]):
        titles[i, topk] = vec[i, topk]
    
    return [', '.join(title) for title in vectorizer.inverse_transform(titles)]
    

def show_clusters(is_system) -> None:
    """
    Show clusters, obtained from train multiwoz with DGAC clustering. Scatter plot with utterances appearing on hover.
    Also assigns names to clusters using TF-IDF and saves them to clust-data directory as .json files.

    Params
    ------
        is_system: bool, flag, indicating what part of clusters to visualize, user of system
    """

    # load necessary data
    clusterer: Clusters = pickle.load(open('clust-data/dgac_clusterer.pickle', 'rb'))
    X = np.load('clust-data/sentence_embeddings.npy')
    speaker = np.array(json.load(open('clust-data/speaker.json', 'r')))

    mask = (speaker == is_system)
        
    projected = pd.DataFrame(TSNE().fit_transform(X[mask]))

    # get names of clusters via tf-idf
    texts = [txt for i, txt in enumerate(json.load(open('clust-data/utterances.json', 'r'))) if speaker[i] == is_system]
    labels = clusterer.labels[speaker == is_system] - is_system * clusterer.n_clusters // 2
    cluster_names = assign_cluster_names(labels, texts)
    
    apndx = 'system' if is_system else 'user'
    json.dump(cluster_names, open(f'clust-data/cluster-tfidf-names-{apndx}.json', 'w'))

    projected['clust_name'] = '-'
    for i in range(clusterer.n_clusters // 2):
        projected.loc[labels == i, 'clust_name'] = cluster_names[i]
    
    # visualize
    projected['label'] = labels
    projected['text'] = texts
    fig = px.scatter(
        projected, x=0, y=1, color='label',
        hover_name='clust_name',
        hover_data={'label': True, 'text': True},
        title=f'{apndx} utterances'
    )

    fig.update_layout()
    fig.show()


def read_csv(path) -> list[str]:
    """Read text from .csv file with two columns: index, text. This function is necessary because of how weirdly pandas saves texts."""
    res = []
    with open(path, 'r') as f:
        f.readline() # row with column names
        for line in f.readlines():
            utter = ','.join(line.split(',')[1:])
            res.append(utter.replace('"', '').strip())
    return res


def get_dialogue(i, name) -> list[str]:
    original = read_csv('aug-data/original.csv')
    rle = json.load(open('aug-data/rle.json', 'r'))
    start = sum(rle[:i])
    end = start + rle[i]
    return original[start:end]


def show_augmented(i, name) -> None:
    """
    Show difference between original dialogue and augmented.
    
    Params
    ------
    - i: int, index of dialogue to construct from aug-data/utterances.json
    - name: {'original', 'clare', 'embedding', 'checklist'}, augmentation method
    """
    original = read_csv('aug-data/original.csv')
    speaker = np.array(json.load(open('clust-data/speaker.json', 'r')))
    rle = json.load(open('aug-data/rle.json', 'r'))

    augmented = read_csv(f'aug-data/{name}.csv')
    start = sum(rle[:i])
    end = start + rle[i]
    speaker_alias = "AB"
    orig = '\n'.join([f'[{speaker_alias[j]}] {ut}' for j, ut in zip(speaker[start:end], original[start:end])])
    aug = '\n'.join([f'[{speaker_alias[j]}] {ut}' for j, ut in zip(speaker[start:end], augmented[start:end])])

    display(Markdown(Redlines(orig, aug).output_markdown))


def show_similarities(i, name, func):
    """
    Show difference between original dialogue and augmented with cluster name provided for each utterance and dialogues similarity.
    
    Params
    ------
    - i: int, index of dialogue to construct from aug-data/utterances.json
    - name: {'original', 'clare', 'embedding', 'checklist'}, augmentation method
    - func: similarity function over bag of nodes vectorization 
    """

    # load texts
    rle = json.load(open('aug-data/rle.json', 'r'))
    start = sum(rle[:i])
    end = start + rle[i]
    orig_txt = read_csv(f'aug-data/original.csv')[start:end]
    aug_txt = read_csv(f'aug-data/{name}.csv')[start:end]
    speaker = json.load(open('aug-data/speaker.json', 'r'))[start:end]

    # load cluster labels
    orig_labels = json.load(open(f'aug-data/clust-labels-original.json'))[start:end]
    aug_labels = json.load(open(f'aug-data/clust-labels-{name}.json'))[start:end]

    # load cluster names
    names = json.load(open(f'clust-data/cluster-tfidf-names-user.json'))
    names.extend(json.load(open(f'clust-data/cluster-tfidf-names-system.json')))

    # parse to redlines markdown 
    orig = []
    aug = []
    speaker_alias = "AB"
    for j in range(rle[i]):
        o_lab = orig_labels[j]
        a_lab = aug_labels[j]
        orig.append(f"""[{speaker_alias[speaker[j]]}] [label: {o_lab}] [name: {names[o_lab]}] {orig_txt[j]}""")
        aug.append(f'[{speaker_alias[speaker[j]]}] [label: {a_lab}] [name: {names[a_lab]}] {aug_txt[j]}')

    # load vectorizations
    orig_vecs = np.load(f'aug-data/vectors-original.npy')[i]
    aug_vecs = np.load(f'aug-data/vectors-{name}.npy')[i]
    print('similarity:', func(orig_vecs, aug_vecs))

    mkdown = Redlines('\n'.join(orig), '\n'.join(aug)).output_markdown
    # print(mkdown)
    display(Markdown(mkdown))