import torch
import torch.nn.functional as F


def contrastive_loss(scores, name):
    if name == 'contrastive_symmetric':
        fn = contrastive_symmetric
    if name == 'contrastive_cross':
        fn = contrastive_cross
    if name == 'contrastive_bce':
        fn = contrastive_bce
    
    return fn(scores)


def contrastive_symmetric(scores):
    batch_size = scores.shape[0]
    targets = torch.eye(batch_size, device=scores.device)
    loss_1 = F.cross_entropy(scores, targets, reduction='mean')
    loss_2 = F.cross_entropy(scores.T, targets, reduction='mean')
    loss = loss_1 + loss_2
    return loss


def contrastive_cross(scores):
    scores = scores.exp()
    pos_scores = scores.diag()
    neg_scores1 = scores.sum(dim=0)
    neg_scores2 = scores.sum(dim=1)
    loss = (pos_scores / (neg_scores1 + neg_scores2 - pos_scores)).log().neg().sum()
    return loss


def contrastive_bce(scores):
    batch_size = scores.shape[0]
    targets = torch.eye(batch_size, device=scores.device)
    loss = F.binary_cross_entropy_with_logits(scores, targets, reduction='mean')
    return loss
