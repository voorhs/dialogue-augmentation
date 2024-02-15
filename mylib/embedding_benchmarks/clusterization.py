from sklearn.cluster import KMeans
import torch
from torchmetrics.functional.classification import multiclass_confusion_matrix
from torchmetrics.functional.clustering import v_measure_score


def all_clusterization_metrics(X_train, Y_train, X_val, Y_val):
    """
    Computes:
    - purity
    - v-measure
    
    X_train: embedding tensor (N, d)
    Y_train: targets (N, k) with zeros and ones
    """

    n_classes = Y_val.shape[1]
    kmeans = KMeans(n_classes, random_state=0, n_init=10)
    pred = kmeans.fit(X_train.numpy()).predict(X_val.numpy())
    pred = torch.from_numpy(pred)
    true = torch.argmax(Y_val, dim=1)
    
    return {
        'cluster_purity': purity(true, pred, n_classes),
        'cluster_v_measure': vmeasure(true, pred, n_classes)
    }

def purity(true, pred, n_classes):
    matrix = multiclass_confusion_matrix(pred, true, n_classes)
    return torch.sum(torch.argmax(matrix, dim=0)) / torch.sum(matrix)

def vmeasure(true, pred, n_classes):
    return v_measure_score(pred, true, n_classes)
