from sklearn.cluster import KMeans
import torch
from torchmetrics.functional.classification import multiclass_confusion_matrix
from torchmetrics.functional.clustering import v_measure_score

from .utils import prepare_data


def all_clusterization_metrics(train_dataset, val_dataset):
    """
    Computes:
    - purity
    - v-measure
    
    train_dataset: tuple of embedding matrix (N, d) and targets (N, k)
    val_dataset: same
    """
    X_train, _, X_val, Y_val = prepare_data(train_dataset, val_dataset)

    n_classes = Y_val.shape[1]
    kmeans = KMeans(n_classes, random_state=0)
    pred = kmeans.fit(X_train.numpy()).predict(X_val.numpy())
    true = torch.argmax(Y_val, dim=1)
    
    return {
        'cluster_purity': purity(true, pred, n_classes),
        'cluster_v_measure': vmeasure(true, pred, n_classes)
    }

def purity(true, pred, n_classes):
    matrix = multiclass_confusion_matrix(pred, true, n_classes)
    return torch.sum(torch.argmax(matrix, dim=0)) / torch.sum(matrix)

def vmeasure(true, pred):
    return v_measure_score(pred, true)
