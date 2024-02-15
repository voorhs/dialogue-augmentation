from typing import Literal

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


def prepare_data(train_dataset, val_dataset):
    X_train = torch.stack([emb for emb, _ in train_dataset], dim=0)
    Y_train = torch.stack([tar for _, tar in train_dataset], dim=0)
    X_val = torch.stack([emb for emb, _ in val_dataset], dim=0)
    Y_val = torch.stack([tar for _, tar in val_dataset], dim=0)
    
    X_train = F.normalize(X_train, dim=1)
    X_val = F.normalize(X_val, dim=1)

    return X_train, Y_train, X_val, Y_val


class KNNClassifier:
    def __init__(self, n_neighbors, metric: Literal['euclidean', 'cosine'] = 'cosine'):
        self.finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric=metric)

    def fit(self, X, y):
        self.finder.fit(X)
        self.labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        y = self.labels[indices]
        N = self.labels.size
        y += (N * np.arange(y.shape[0]))[:, None]
        ans = np.bincount(
            y.ravel(),
            minlength=N*y.shape[0],
        ).reshape(-1, N)

        return ans.argmax(axis=1)

    def kneighbors(self, X, return_distance=False):
        return self.finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    def __init__(self, batch_size, n_neighbors, metric='euclidean'):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            metric=metric,
        )
        self.batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        assert self.batch_size <= X.shape[0]

        split_indices = [index for index in range(self.batch_size, X.shape[0], self.batch_size)]
        batches = np.vsplit(X, split_indices)

        distances = []
        indices = []
        for batch in batches:
            dist, ind = self.finder.kneighbors(batch, return_distance=True)
            distances.append(dist)
            indices.append(ind)

        if return_distance:
            return np.vstack(distances), np.vstack(indices)

        return np.vstack(indices)
