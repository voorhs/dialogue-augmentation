import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from datasets import load_from_disk, disable_caching, concatenate_datasets
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm

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
    sns.scatterplot(df, x=0, y=1, hue="services", style="aug", s=2)
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

if __name__ == "__main__":
    paths = {
        "BERT": "bert-filtered",
        "BERT-trained": "bert-trained-filtered",
        "BGE": "bge-filtered",
        "BGE-trained": "bge-trained-filtered",
        "DSE": "dse-filtered",
        "DSE-trained": "dse-trained-filtered",
        "ANCE": "ance-filtered",
        "DPR": "dpr-filtered",
        "SFR": "sfr-embedded",
        "RetroMAE": "retromae-filtered",
        "RetroMAE-trained": "retromae-trained-filtered",
    }

    for model, ds_path in paths.items():
        dataset = load_from_disk(f"data/scatter-analysis/{ds_path}/sgd")
        dataset = oh_to_str(dataset)
        dataset = add_const_column(dataset, "aug", "orig")
        df = get_tsne_df(dataset)
        dataset_augs = {}
        df_augs = {}
        for aug in ["insert", "replace", "prune", "shuffle"]:
            dataset_aug = load_from_disk(f"data/scatter-analysis/{ds_path}/sgd-{aug}")
            dataset_aug = oh_to_str(dataset_aug)
            dataset_aug = add_const_column(dataset_aug, "aug", aug)
            
            df_aug = get_pca_df(dataset, dataset_aug)
            visualize_pca(df_aug, f"orig + {aug}", model)

            # df_aug = get_tsne_df(concatenate_datasets([dataset, dataset_aug]))
            # visualize_tsne(df, df_aug, "orig", f"orig + {aug}", model)

            # cosine_analysis(dataset, dataset_aug, model, aug)
