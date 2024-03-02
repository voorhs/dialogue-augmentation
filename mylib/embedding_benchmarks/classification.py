import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

from .utils import BatchedKNNClassifier


def all_classification_metrics(X_train, Y_train, X_val, Y_val, multilabel):
    """
    Computes:
    - logreg accuracy
    - logreg F1
    - 1nn accuracy
    - 1nn F1
    - 5nn accuracy
    - 5nn F1
    - 10nn accuracy
    - 10nn F1
    - 50nn accuracy
    - 50nn F1
    
    X_train: embedding tensor (N, d)
    Y_train: targets (N, k) with zeros and ones
    """
    
    X_train = X_train.numpy()
    X_val = X_val.numpy()
    Y_train = Y_train.numpy()
    Y_val = Y_val.numpy()

    if not multilabel:
        Y_train = np.argmax(Y_train, axis=1)
        Y_val = np.argmax(Y_val, axis=1)
    
    if not multilabel:
        # logreg
        logreg = LogisticRegression(random_state=0)
        pred_logreg = logreg.fit(X_train, Y_train).predict(X_val)
        
        res = {
            'logreg_accuracy': accuracy_score(Y_val, pred_logreg),
            'logreg_f1': f1_score(Y_val, pred_logreg, average='macro'),
        } 
    else:
        mlp = MLPClassifier(hidden_layer_sizes=(), random_state=0)
        pred_mlp = mlp.fit(X_train, Y_train).predict(X_val)
        res = {
            'mlp_f1': f1_score(Y_val, pred_mlp, average='macro'),
        }

    if not multilabel:
        # knn
        batch_size = min(256, X_val.shape[0])
        knn = BatchedKNNClassifier(n_neighbors=50, batch_size=batch_size)
        knn.fit(X_train, Y_train)
        k_distances, k_indices = knn.kneighbors(X_val, return_distance=True)

        for k in [1, 5, 10, 50]:
            y_pred = knn._predict_precomputed(k_indices[:, :k], k_distances[:, :k])
            res[f'{k}nn_accuracy'] = accuracy_score(Y_val, y_pred)
            res[f'{k}nn_f1'] = f1_score(Y_val, y_pred, average='macro')
        
    return res
