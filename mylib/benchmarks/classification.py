import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score, re
from .utils import prepare_data


def all_classification_metrics(train_dataset, val_dataset):
    """
    Computes:
    - logreg accuracy
    - logreg F1
    - knn accuracy
    - knn F1
    
    train_dataset: tuple of embedding matrix (N, d) and targets (N, k)
    val_dataset: same
    """
    X_train, Y_train, X_val, Y_val = prepare_data(train_dataset, val_dataset)

    logreg = LogisticRegression(random_state=0)
    true = np.argmax(Y_val.numpy(), axis=1)
    pred_proba = logreg.fit(X_train.numpy(), true).predict_proba(X_val)
    pred_labels = (pred_proba > 0.5).astype(np.int_)

#     return {
#         'logreg_accuracy'
#     } 

# def
