import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalHitRate, RetrievalNormalizedDCG, RetrievalRecall
from sklearn.metrics import average_precision_score

from .utils import prepare_data


def MAP_ranking(train_dataset, val_dataset):
    """
    train_dataset: tuple of embedding matrix (N, d) and targets (N, k)
    val_dataset: same
    """
    
    X_train, Y_train, X_val, Y_val = prepare_data(train_dataset, val_dataset)
    
    avg_scores = []
    for x_val, y_val in zip(X_val, Y_val):
        # indicates that train sample and val sample has at least one common class
        labels = (Y_train @ y_val) > 0

        # cosine similarities
        scores = X_train @ x_val

        # AP is an area under rectangularly interpolated PR curve
        # for detailed explanation see my guide:
        # https://nbviewer.org/github/voorhs/ml-practice/blob/main/average-precision-comparision.ipynb 
        avg_scores.append(average_precision_score(labels.numpy(), scores.numpy()))
    
    return sum(avg_scores) / len(avg_scores)
    

def all_retrieval_metrics(train_dataset, val_dataset):
    """
    Computes:
    - MAP
    - hit rate
    - recall
    - ndcg

    train_dataset: tuple of embedding matrix (N, d) and targets (N, k)
    val_dataset: same
    """
    
    X_train, Y_train, X_val, Y_val = prepare_data(train_dataset, val_dataset)

    preds = X_val @ X_train.T
    targets = Y_val @ Y_train.T

    train_size = X_train.shape[0]
    val_size = X_val.shape[0]    
    indexes = torch.arange(val_size).unsqueeze(1).expand(val_size, train_size)

    res = {}
    for metric in [RetrievalMAP, RetrievalHitRate, RetrievalNormalizedDCG, RetrievalRecall]:
        for k in [10, 100, None]:
            metric_name = metric.__name__
            if k is not None:
                metric_name = metric_name + f'_{k}'
            
            res[metric_name] = metric(top_k=k).compute(preds, targets, indexes)

    return res