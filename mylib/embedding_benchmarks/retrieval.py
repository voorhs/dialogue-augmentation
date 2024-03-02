import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalHitRate, RetrievalNormalizedDCG, RetrievalRecall
    

def all_retrieval_metrics(X_train, Y_train, X_val, Y_val, multilabel=None):
    """
    Computes:
    - MAP
    - hit rate
    - recall
    - ndcg

    X_train: embedding tensor (N, d)
    Y_train: targets (N, k) with zeros and ones
    """

    preds = X_val @ X_train.T
    targets = (Y_val @ Y_train.T) > 0

    train_size = X_train.shape[0]
    val_size = X_val.shape[0]    
    indexes = torch.arange(val_size).unsqueeze(1).expand(val_size, train_size)

    res = {}
    for metric in [RetrievalMAP, RetrievalHitRate, RetrievalNormalizedDCG, RetrievalRecall]:
        for k in [1, 10, 100, None]:
            metric_name = metric.__name__
            if k is not None:
                metric_name = metric_name + f'_{k}'
            
            res[metric_name] = metric(top_k=k)(preds, targets, indexes)

    return res