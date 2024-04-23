import os

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
from sklearn.neighbors import KNeighborsClassifier

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

def get_tsne_df(dataset, projector="tsne", seed=0):
    embeddings = np.array(dataset["embedding"])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    if projector == "tsne":
        np.random.seed(seed)
        init = np.random.randn(len(embeddings), 2)
        projected_embeddings = TSNE(init=init, random_state=seed).fit_transform(
            embeddings
        )
    else:
        raise ValueError("unknown projector")

    projected = pd.DataFrame(projected_embeddings)
    projected["aug"] = dataset["aug"]
    projected["services"] = dataset["services"]

    return projected


def get_pca_df(ds1, ds2, seed=0):
    embeddings1 = np.array(ds1["embedding"])
    embeddings2 = np.array(ds2["embedding"])
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


def visualize_pca(df, title, model):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(df[df['aug'] == 'orig'], x=0, y=1, hue="services", s=2, ax=ax[0])
    sns.scatterplot(df, x=0, y=1, hue="services", style="aug", s=2, ax=ax[1])
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.1, 1.05))

    folder = f"figures/{model}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(os.path.join(folder, f"{title}-pca.svg"), bbox_inches="tight")
    plt.close()


def visualize_tsne(df1, df2, title1, title2, suptitle):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(
        df1, x=0, y=1, hue="services", style="aug", ax=ax[0], legend=False, s=2
    )
    sns.scatterplot(df2, x=0, y=1, hue="services", style="aug", ax=ax[1], s=2)
    ax[0].set_title(title1)
    ax[1].set_title(title2)
    ax[1].legend(bbox_to_anchor=(1.1, 1.05))

    fig.suptitle(suptitle)

    folder = f"figures/{suptitle}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig.savefig(os.path.join(folder, f"{title1}_{title2}.svg"), bbox_inches="tight")
    plt.close()


def add_const_column(dataset, name, value):
    dataset = dataset.add_column(name, [value] * len(dataset))
    return dataset


def cosine_analysis(ds_1, ds_2, model, aug):
    df_1 = ds_1.to_pandas()
    df_2 = ds_2.to_pandas()
    df_joined = df_1.join(
        df_2, on="idx_within_source", how="inner", rsuffix="_augmented"
    )
    orig_emb = np.stack(df_joined["embedding"])
    aug_emb = np.stack(df_joined["embedding_augmented"])
    df_joined["cos"] = (orig_emb * aug_emb).sum(axis=1) / (
        norm(orig_emb, axis=1) * norm(aug_emb, axis=1)
    )

    mean = df_joined["cos"].mean()
    std = df_joined["cos"].std()

    n_services = len(sgd_labels)

    sns.histplot(df_joined, x="cos")
    plt.title(f"{model}/{aug}/All [mean={mean:.3f}, std={std:.3f}]")

    folder = f"figures/{model}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    fpath = os.path.join(folder, f"{aug}-all.svg")

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
    fpath = os.path.join(folder, f"{aug}-grid.svg")

    plt.savefig(fpath, bbox_inches="tight")
    plt.close()

def aug_prediction(ds_1, ds_2, model, aug, seed=0):
    df_1 = ds_1.to_pandas()
    df_2 = ds_2.to_pandas()
    df_joined = df_1.join(df_2, on="idx_within_source", how="inner", rsuffix="_augmented")
    
    emb_orig = np.stack(df_joined['embedding'])
    emb_aug = np.stack(df_joined['embedding_augmented'])

    # emb_orig /= np.linalg.norm(emb_orig, axis=1, keepdims=True)
    # emb_aug /= np.linalg.norm(emb_aug, axis=1, keepdims=True)
    
    np.random.seed(seed)
    choose_orig = np.random.randn(len(df_joined)) > 0
    X = emb_aug.copy()
    X[choose_orig] = emb_orig[choose_orig]

    y = choose_orig.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_accuracy = logreg.score(X_test, y_test)

    knn = KNeighborsClassifier(metric='cosine')
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    
    folder = f"figures/{model}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    fpath = os.path.join(folder, f"{aug}-prediction.txt")

    open(fpath, 'w').write(f'{logreg_accuracy=:.4f}\n{knn_accuracy=:.4f}')


def aug_prediction_all(ds_list, model, seed=0):
    df_list = [ds.to_pandas() for ds in ds_list]

    df_joined = df_list[0].join(df_list[1], on="idx_within_source", how="inner", rsuffix="_1")

    if len(df_list) > 1:
        for i in range(2, len(df_list)):
            df_joined = df_joined.join(df_list[i], on="idx_within_source", how="inner", rsuffix=f"_{i}")
    df_joined.rename(columns={'embedding': 'embedding_0'}, inplace=True)
    
    np.random.seed(seed)
    y = np.random.choice(len(df_list), size=len(df_joined))
    X = [df_joined.loc[i, f'embedding_{aug}'] for i, aug in enumerate(y)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_accuracy = logreg.score(X_test, y_test)

    knn = KNeighborsClassifier(metric='cosine')
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    
    folder = f"figures/{model}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    fpath = os.path.join(folder, "prediction-all.txt")

    open(fpath, 'w').write(f'{logreg_accuracy=:.4f}\n{knn_accuracy=:.4f}')


if __name__ == "__main__":
    paths = {
        "BERT": "bert-filtered",
        "BERT-trained-1": "bert-traned-1-filtered",
        "BERT-trained-2": "bert-traned-2-filtered",
        "BERT-trained-3": "bert-traned-3-filtered",
        "BERT-trained-4": "bert-traned-4-filtered",
        "BGE": "bge-filtered",
        "BGE-trained-1": "bge-trained-1-filtered",
        "BGE-trained-2": "bge-trained-2-filtered",
        "BGE-trained-3": "bge-trained-3-filtered",
        "BGE-trained-4": "bge-trained-4-filtered",
        "DSE": "dse-filtered",
        "DSE-trained-1": "dse-trained-1-filtered",
        "DSE-trained-2": "dse-trained-2-filtered",
        "DSE-trained-3": "dse-trained-3-filtered",
        "DSE-trained-4": "dse-trained-4-filtered",
        "ANCE": "ance-filtered",
        "DPR": "dpr-filtered",
        "SFR": "sfr-embedded",
        "RetroMAE": "retromae-filtered",
        "RetroMAE-trained-1": "retromae-trained-1-filtered",
        "RetroMAE-trained-2": "retromae-trained-2-filtered",
        "RetroMAE-trained-3": "retromae-trained-3-filtered",
        "RetroMAE-trained-4": "retromae-trained-4-filtered",
        
    }

    for model, ds_path in paths.items():
        dataset = {}
        dataset['orig'] = load_from_disk(f"data/scatter-analysis/{ds_path}/sgd")
        dataset['orig'] = oh_to_str(dataset['orig'])
        dataset['orig'] = add_const_column(dataset['orig'], "aug", "orig")
        df = {}
        df['orig'] = get_tsne_df(dataset['orig'])
        for aug in ["insert", "replace", "prune", "shuffle"]:
            dataset[aug] = load_from_disk(f"data/scatter-analysis/{ds_path}/sgd-{aug}")
            dataset[aug] = oh_to_str(dataset[aug])
            dataset[aug] = add_const_column(dataset[aug], "aug", aug)
            
            df[aug] = get_pca_df(dataset['orig'], dataset[aug])
            visualize_pca(df[aug], f"orig + {aug}", model)

            df[aug] = get_tsne_df(concatenate_datasets([dataset, dataset[aug]]))
            visualize_tsne(df['orig'], df[aug], "orig", f"orig + {aug}", model)

            cosine_analysis(dataset['orig'], dataset[aug], model, aug)

            aug_prediction(dataset['orig'], dataset[aug], model, aug)

        aug_prediction_all(list(dataset.values()), model)
