from sklearn.cluster import KMeans
import torch
from torchmetrics.functional.classification import multiclass_confusion_matrix
from torchmetrics.functional.clustering import v_measure_score


def all_clusterization_metrics(X_train, Y_train, X_val, Y_val, multilabel=False):
    """
    Computes:
    - purity
    - v-measure
    
    X_train: embedding tensor (N, d)
    Y_train: targets (N, k) with zeros and ones
    """
    
    if multilabel:
        return {}

    X_train = X_train.numpy()
    X_val = X_val.numpy()
    
    n_classes = Y_val.shape[1]
    kmeans = KMeans(n_classes, random_state=0, n_init=10)
    
    pred = torch.from_numpy(kmeans.fit(X_train).predict(X_val))
    true = torch.argmax(Y_val, dim=1)

    pred_train = torch.from_numpy(kmeans.predict(X_train))
    true_train = torch.argmax(Y_train, dim=1)
    
    return {
        'cluster_purity_val': purity(true, pred, n_classes),
        'cluster_v_measure_val': vmeasure(true, pred, n_classes),
        'cluster_purity_train': purity(true_train, pred_train, n_classes),
        'cluster_v_measure_train': vmeasure(true_train, pred_train, n_classes)
    }

def purity(true, pred, n_classes):
    matrix = multiclass_confusion_matrix(pred, true, n_classes)
    return torch.sum(torch.max(matrix, dim=0)[0]) / torch.sum(matrix)

def vmeasure(true, pred, n_classes):
    return v_measure_score(pred, true, n_classes)
