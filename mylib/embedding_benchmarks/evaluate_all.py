from warnings import catch_warnings, filterwarnings

from .retrieval import all_retrieval_metrics
from .clusterization import all_clusterization_metrics
from .classification import all_classification_metrics
from .utils import prepare_data


def all_embedding_metrics(train_dataset, val_dataset, multilabel=False):
    X_train, Y_train, X_val, Y_val = prepare_data(train_dataset, val_dataset)

    with catch_warnings():
        filterwarnings('ignore')
        clf_metrics = all_classification_metrics(X_train, Y_train, X_val, Y_val, multilabel)
        retr_metrics = all_retrieval_metrics(X_train, Y_train, X_val, Y_val, multilabel)
        clust_metrics = all_clusterization_metrics(X_train, Y_train, X_val, Y_val, multilabel)
    
    return dict(**clf_metrics, **retr_metrics, **clust_metrics)
