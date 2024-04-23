import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import concatenate_datasets, disable_caching, load_from_disk
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from mylib.embedding_benchmarks.utils import BatchedKNNClassifier

sns.set_style("whitegrid")


disable_caching()

# UTILS

sgd_labels = [
    "Banks",
    "Buses",
    "Calendar",
    "Events",
    "Flights",
    "Homes",
    "Hotels",
    "Media",
    "Movies",
    "Music",
    "RentalCars",
    "Restaurants",
    "RideSharing",
    "Services",
]


def get_labels(onehot):
    index = np.nonzero(onehot)[0][0]
    return {"services": sgd_labels[index]}


def oh_to_str(dataset):
    return dataset.map(get_labels, input_columns="services")


def get_tsne_df(dataset, key, seed=0):
    embeddings = np.array(dataset[key])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.random.seed(seed)
    init = np.random.randn(len(embeddings), 2)
    projected_embeddings = TSNE(init=init, random_state=seed).fit_transform(
        embeddings
    )


    projected = pd.DataFrame(projected_embeddings)
    projected["aug"] = dataset["aug"]
    projected["services"] = dataset["services"]

    return projected


def get_pca_df(ds1, ds2, key, seed=0):
    embeddings1 = np.array(ds1[key])
    embeddings2 = np.array(ds2[key])
    embeddings1 /= np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 /= np.linalg.norm(embeddings2, axis=1, keepdims=True)

    np.random.seed(seed)
    pca = PCA(n_components=2)
    projected_embeddings1 = pca.fit_transform(embeddings1)
    projected_embeddings2 = pca.transform(embeddings2)

    projected1 = pd.DataFrame(projected_embeddings1)
    projected1["aug"] = ds1["aug"]
    projected1["services"] = ds1["services"]

    projected2 = pd.DataFrame(projected_embeddings2)
    projected2["aug"] = ds2["aug"]
    projected2["services"] = ds2["services"]

    return pd.concat([projected1, projected2], axis=0)


def visualize_pca(df, model, aug):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(df[df['aug'] == 'orig'], x=0, y=1, hue="services", s=2, ax=ax[0])
    sns.scatterplot(df, x=0, y=1, hue="services", style="aug", s=2, ax=ax[1])
    fig.suptitle(model)
    ax[0].set_title('orig')
    ax[1].set_title(aug)
    ax[1].legend(bbox_to_anchor=(1.1, 1.05))

    folder = f"figures/{model}/pca"
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(os.path.join(folder, f"{aug}.svg"), bbox_inches="tight")
    plt.close()


def visualize_tsne(df1, df2, model, aug):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(df1, x=0, y=1, hue="services", ax=ax[0], legend=False, s=2)
    sns.scatterplot(df2, x=0, y=1, hue="services", style="aug", ax=ax[1], s=2)
    ax[0].set_title('orig')
    ax[1].set_title(aug)
    ax[1].legend(bbox_to_anchor=(1.1, 1.05))

    fig.suptitle(model)

    folder = f"figures/{model}/tsne"
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig.savefig(os.path.join(folder, f"{aug}.svg"), bbox_inches="tight")
    plt.close()


def add_const_column(dataset, name, value):
    dataset = dataset.add_column(name, [value] * len(dataset))
    return dataset


def cosine_analysis(ds_1, ds_2, model, aug, key):
    df_1 = ds_1.to_pandas()
    df_2 = ds_2.to_pandas()
    df_joined = df_1.join(
        df_2, on="idx_within_source", how="inner", rsuffix="_augmented"
    )
    orig_emb = np.stack(df_joined[key])
    aug_emb = np.stack(df_joined[f"{key}_augmented"])
    df_joined["cos"] = (orig_emb * aug_emb).sum(axis=1) / (
        norm(orig_emb, axis=1) * norm(aug_emb, axis=1)
    )

    mean = df_joined["cos"].mean()
    std = df_joined["cos"].std()

    n_services = len(sgd_labels)

    sns.histplot(df_joined, x="cos")
    plt.title(f"{model}/{aug}/All [mean={mean:.3f}, std={std:.3f}]")

    folder = f"figures/{model}/cosine"
    if not os.path.exists(folder):
        os.makedirs(folder)
    fpath = os.path.join(folder, f"heap-{aug}.svg")

    plt.savefig(fpath, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1, n_services, figsize=(5 * n_services, 5), sharey=True)

    for i, service in enumerate(sgd_labels):
        sub_df = df_joined[df_joined.services == service]
        mean = sub_df["cos"].mean()
        std = sub_df["cos"].std()
        sns.histplot(sub_df, x="cos", ax=ax[i])
        ax[i].set_title(f"{service} [mean={mean:.3f}, std={std:.3f}]")

    folder = f"figures/{model}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    fpath = os.path.join(folder, f"classwise-{aug}.svg")

    plt.savefig(fpath, bbox_inches="tight")
    plt.close()


def aug_prediction(ds_list, model, aug, key, seed=0):
    df_list = [ds.to_pandas() for ds in ds_list]

    df_joined = df_list[0].join(df_list[1], on="idx_within_source", how="inner", rsuffix="_1")

    if len(df_list) > 1:
        for i in range(2, len(df_list)):
            df_joined = df_joined.join(df_list[i], on="idx_within_source", how="inner", rsuffix=f"_{i}")
    df_joined.rename(columns={key: f'{key}_0'}, inplace=True)

    np.random.seed(seed)
    y = np.random.choice(len(df_list), size=len(df_joined))
    X = np.stack([df_joined.loc[i, f'{key}_{i_aug}'] for i, i_aug in enumerate(y)], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    dump_scores(X_train, X_test, y_train, y_test, model, aug)


def dump_scores(X_train, X_test, y_train, y_test, model, aug):
    res = {}
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    res['logreg'] = classification_report(y_test, y_pred, output_dict=True)

    batch_size = min(256, len(X_test))
    knn = BatchedKNNClassifier(n_neighbors=50, batch_size=batch_size)
    knn.fit(X_train, y_train)
    k_distances, k_indices = knn.kneighbors(X_test, return_distance=True)

    for k in [1, 3, 5, 10, 20]:
        y_pred = knn._predict_precomputed(k_indices[:, :k], k_distances[:, :k])
        res[f'{k}nn'] = classification_report(y_test, y_pred, output_dict=True)
    
    folder = f"figures/{model}/aug-prediction"
    if not os.path.exists(folder):
        os.makedirs(folder)
    fpath = os.path.join(folder, f"{aug}.json")

    json.dump(res, open(fpath, 'w'))
    

if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('--key', default='embedding')
    args = ap.parse_args()

    paths = {
        "BERT": "BERT-filtered",
        "BERT-trained-0": "BERT-trained-0-filtered",
        "BERT-trained-1": "BERT-trained-1-filtered",
        "BERT-trained-2": "BERT-trained-2-filtered",
        "BERT-trained-3": "BERT-trained-3-filtered",
        "BERT-trained-4": "BERT-trained-4-filtered",
        "BGE": "BGE-filtered",
        "BGE-trained-0": "BGE-trained-0-filtered",
        "BGE-trained-1": "BGE-trained-1-filtered",
        "BGE-trained-2": "BGE-trained-2-filtered",
        "BGE-trained-3": "BGE-trained-3-filtered",
        "BGE-trained-4": "BGE-trained-4-filtered",
        "DSE": "DSE-filtered",
        "DSE-trained-0": "DSE-trained-0-filtered",
        "DSE-trained-1": "DSE-trained-1-filtered",
        "DSE-trained-2": "DSE-trained-2-filtered",
        "DSE-trained-3": "DSE-trained-3-filtered",
        "DSE-trained-4": "DSE-trained-4-filtered",
        # "ANCE": "ance-filtered",
        # "DPR": "dpr-filtered",
        # "SFR": "sfr-embedded",
        "RetroMAE": "RetroMAE-filtered",
        "RetroMAE-trained-0": "RetroMAE-trained-0-filtered",
        "RetroMAE-trained-1": "RetroMAE-trained-1-filtered",
        "RetroMAE-trained-2": "RetroMAE-trained-2-filtered",
        "RetroMAE-trained-3": "RetroMAE-trained-3-filtered",
        "RetroMAE-trained-4": "RetroMAE-trained-4-filtered"   
    }
    aug_list = ["insert", "replace", "prune", "shuffle"]

    for model, ds_path in paths.items():
        model_key = f'{model}-{args.key}'
        dataset = {}
        dataset['orig'] = load_from_disk(f"data/scatter-analysis/{ds_path}/sgd")
        dataset['orig'] = oh_to_str(dataset['orig'])
        dataset['orig'] = add_const_column(dataset['orig'], "aug", "orig")
        df = {}
        df['orig'] = get_tsne_df(dataset['orig'], key=args.key)
        for aug in aug_list:
            dataset[aug] = load_from_disk(f"data/scatter-analysis/{ds_path}/sgd-{aug}")
            dataset[aug] = oh_to_str(dataset[aug])
            dataset[aug] = add_const_column(dataset[aug], "aug", aug)
            
            df[aug] = get_pca_df(dataset['orig'], dataset[aug], key=args.key)
            visualize_pca(df[aug], model_key, aug)

            df[aug] = get_tsne_df(concatenate_datasets([dataset['orig'], dataset[aug]]), key=args.key)
            visualize_tsne(df['orig'], df[aug], model_key, aug)

            cosine_analysis(dataset['orig'], dataset[aug], model_key, aug, key=args.key)
            aug_prediction([dataset['orig'], dataset[aug]], model_key, aug, key=args.key)

        aug_prediction(list(dataset.values()), model_key, 'all', key=args.key)

        df['all'] = pd.concat(list(df.values()), axis=0)
        visualize_pca(df['all'], model_key, 'all')

        # dataset['all-augs'] = concatenate_datasets([dataset[aug] for aug in aug_list])
        # cosine_analysis(dataset['orig'], dataset['all-augs'], model_key, 'all', key=args.key)
