import numpy as np
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch
from typing import Literal
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, average_precision_score
from scipy.stats import pearsonr
from .generic import BaseLearnerConfig, BaseLearner
from sklearn.preprocessing import normalize as sklearn_normalize


@dataclass
class DialogueEncoderLearnerConfig(BaseLearnerConfig):
    path_to_gold_multiwoz_intent_similarities: str = ''
    k: int = 5
    temperature: float = 0.05
    loss: Literal['contrastive', 'ict', 'multiwoz_service_clf'] = 'contrastive'
    finetune_layers: int = 0
    contrastive_train_frac: float = 1.
    multiwoz_train_frac: float = 1.
    multiwoz_val_frac: float = 1.


class DialogueEncoderLearner(BaseLearner):
    def __init__(self, model, config: DialogueEncoderLearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

        # list of (embedding, target) pairs for multiwoz service clf (as validation)
        self.multiwoz_train = []
        self.multiwoz_validation = []

        if self.config.loss == 'multiwoz_service_clf':
            self.clf_head = nn.Linear(self.model.get_hidden_size(), 7)

    def forward(self, batch):
        if self.config.loss == 'contrastive':
            return self._contrastive_step(batch)
        if self.config.loss == 'ict':
            raise NotImplementedError()
        if self.config.loss == 'multiwoz_service_clf':
            raise NotImplementedError()

    def _contrastive_step(self, batch):
        """`batch` is a list of samples from ContrastiveDataset"""
        origs = [sample['orig']['content'] for sample in batch]
        
        # select positives
        points = np.random.uniform(low=0, high=1, size=len(batch))
        counts = np.array([len(sample['pos']) for sample in batch])
        pos_indices = np.floor(points * counts).astype(np.int_)
        
        positives = [sample['pos'][i]['content'] for i, sample in zip(pos_indices, batch)]

        # select hard_negatives
        hard_negatives = []
        hard_negatives_counts = []
        for sample in batch:
            negs = [dia['content'] for dia in sample['neg']]
            hard_negatives.extend(negs)
            hard_negatives_counts.append(len(negs))

        # encode all dialogues
        origs_enc = F.normalize(self.model(origs), dim=1)                   # (B, H)
        positives_enc = F.normalize(self.model(positives), dim=1)           # (B, H)
        hard_negatives_enc = F.normalize(self.model(hard_negatives), dim=1) # (B+, H)

        # pos and neg scores
        pairwise_scores = (origs_enc @ positives_enc.T / self.config.temperature).exp()
        pos_scores = pairwise_scores.diag()
        neg_scores1 = pairwise_scores.sum(dim=0)
        neg_scores2 = pairwise_scores.sum(dim=1)
        
        # hard neg scores
        repeats = torch.tensor(hard_negatives_counts, device=self.model.device)
        origs_enc_repeated = torch.repeat_interleave(origs_enc, repeats=repeats, dim=0)
        _hard_neg_scores = (origs_enc_repeated * hard_negatives_enc / self.config.temperature).sum(dim=1).exp()
        hard_neg_scores = []
        for i, count in enumerate(hard_negatives_counts):
            start = sum(hard_negatives_counts[:i])
            end = start + count
            score = _hard_neg_scores[start:end].sum()
            hard_neg_scores.append(score)
        hard_neg_scores = torch.tensor(hard_neg_scores, device=self.model.device)

        # compute contrastive loss with hard negatives
        loss = (pos_scores / (neg_scores1 + neg_scores2 + hard_neg_scores)).log().neg().sum()
        
        # compute metric: retrieval accuracy
        topk_indicators = [i in top for i, top in enumerate(torch.topk(pairwise_scores, k=self.config.k, dim=1).indices)]
        topk_accuracy = np.mean(topk_indicators)

        return loss, topk_accuracy

    def _multiwoz_service_clf_step(self, batch):
        """`batch` is a list of samples from MultiWOZServiceClfDataset"""
        dialogues = [dia for dia, _ in batch]
        targets = torch.stack([tar for _, tar in batch], dim=0)
        
        embeddings = self.model(dialogues)  # (B, H)
        logits = self.clf_head(embeddings)  # (B, 7)
        
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        metric = multilabel_f1_score(logits, targets, average='macro')

        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.forward(batch)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        self.log(
            name='train_metric',
            value=metric,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        dialogues = [dia for dia, _ in batch]
        targets = torch.stack([tar for _, tar in batch], dim=0).detach().cpu().numpy()
        embeddings = self.model(dialogues).detach().cpu().numpy()
        res = list(zip(embeddings, targets))

        if dataloader_idx == 0:
            self.multiwoz_train.extend(res)
        elif dataloader_idx == 1:
            self.multiwoz_validation.extend(res)
    
    def on_validation_epoch_end(self) -> None:
        corr_metric = get_multiwoz_intent_correlation_score(
            self.multiwoz_train,
            self.multiwoz_validation,
            np.load(self.config.path_to_gold_multiwoz_intent_similarities)
        )

        clf_metric = get_multiwoz_service_clf_score(
            self.multiwoz_train,
            self.multiwoz_validation
        )

        ranking_metric = get_multiwoz_service_ranking_score(
            self.multiwoz_train,
            self.multiwoz_validation
        )

        self.log_dict(
            dictionary={
                'clf_metric': clf_metric,
                'ranking_metric': ranking_metric,
                'corr_metric': corr_metric
            },
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )

        self.multiwoz_train.clear()
        self.multiwoz_validation.clear()

    def configure_optimizers(self):
        optim_groups = self.get_parameter_groups()
        optimizer = AdamW(optim_groups, lr=self.config.lr, betas=self.config.betas)
        def lr_foo(step):
            warmup_steps = self.config.warmup_period
            periodic = self.config.do_periodic_warmup
            
            if warmup_steps is None:
                return 1
            if periodic:
                return (step % warmup_steps + 1) / warmup_steps
            else:
                return (step + 1) / warmup_steps if step < warmup_steps else 1

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1}}


def get_multiwoz_service_clf_score(train_dataset, val_dataset, n_epochs=2):
    # configure model
    clf = MLPClassifier(
        batch_size=32,
        learning_rate_init=5e-4,
        max_iter=n_epochs
    )

    # configure data
    X_train = np.stack([emb for emb, _ in train_dataset], axis=0)
    y_train = np.stack([tar for _, tar in train_dataset], axis=0)
    X_val = np.stack([emb for emb, _ in val_dataset], axis=0)
    y_val = np.stack([tar for _, tar in val_dataset], axis=0)
    
    # train model
    clf.fit(X_train, y_train)

    # score model
    y_pred = clf.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro', zero_division=0)
    
    return score


def get_multiwoz_service_ranking_score(train_dataset, val_dataset):
    # configure data
    X_train = np.stack([emb for emb, _ in train_dataset], axis=0)
    Y_train_raw = np.stack([tar for _, tar in train_dataset], axis=0)
    X_val = np.stack([emb for emb, _ in val_dataset], axis=0)
    Y_val_raw = np.stack([tar for _, tar in val_dataset], axis=0)
    
    X_train = sklearn_normalize(X_train, axis=1)
    X_val = sklearn_normalize(X_val, axis=1)

    avg_scores = []
    for x_val, y_val_raw in zip(X_val, Y_val_raw):
        # indicates that train sample and val sample has at least one common service
        labels = np.sum(Y_train_raw * y_val_raw[None, :], axis=1) > 0

        # cosine similarities
        scores = X_train @ x_val

        avg_scores.append(average_precision_score(labels, scores))
    
    map_score = sum(avg_scores) / len(avg_scores)
    return map_score


def get_multiwoz_intent_correlation_score(train_dataset, val_dataset, gold_intent_similarities):
    X_train = np.stack([emb for emb, _ in train_dataset], axis=0)
    X_val = np.stack([emb for emb, _ in val_dataset], axis=0)

    X_train = sklearn_normalize(X_train, axis=1)
    X_val = sklearn_normalize(X_val, axis=1)

    pred_intent_similatities = X_train @ X_val.T
    corr_score = pearsonr(pred_intent_similatities.flatten(), gold_intent_similarities.flatten()).statistic
    
    return corr_score
