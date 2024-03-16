from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..modeling.pairwise import SimplifiedPairwise, SimplifiedPairwiseModelConfig
from ..utils.modeling.generic import mySentenceTransformer


def _load_pairwise_cat(ckpt_path, device):
    encoder_name = 'aws-ai/dse-bert-base'
    _encoder = mySentenceTransformer(encoder_name)

    model = SimplifiedPairwise(
        model=_encoder,
        config=SimplifiedPairwiseModelConfig(context_size=3)
    ).to(device)
    return model.eval()


class Pruner:
    def __init__(
            self,
            ckpt_path='./logs/comet/pairwise-model/84e24444441141819e9934acbf055f5f/checkpoints/last.ckpt',
            device='cuda',
        ):
        self.model = _load_pairwise_cat(ckpt_path, device)

    def __call__(self, dialogues):
        return [self._cut(self.model, dia) for dia in tqdm(dialogues, desc='cutting dialogues')]
        

    @staticmethod
    def _cut(model: SimplifiedPairwise, dia):
        """drops all clusters except the biggest one. applies transformation only to dialogues with 6 utterances at least"""
        if len(dia) < 6:
            return None, -np.inf

        end = len(dia) // 3
        start = 2

        choices = list(range(start, end+1))
        n_clusters = np.random.choice(choices)

        clusterwise_uts = _cluster(model, dia, n_clusters)
        ids = clusterwise_uts[np.argmax([len(clust) for clust in clusterwise_uts])]
        aug = [dia[i] for i in ids]

        # variations = []
        # for n_clusters in range(start, end+1):
        #     clusterwise_uts = _cluster(model, dia, n_clusters)
        #     ids = clusterwise_uts[np.argmax([len(clust) for clust in clusterwise_uts])]
        #     aug = [dia[i] for i in ids]
        #     score = model.score(aug)
        #     variations.append((aug, score))
        # res, score = max(variations, key=lambda x: x[1])
        
        return aug


@torch.no_grad()
def _cluster(model: SimplifiedPairwise, dia, n_clusters):
    """clusters utterances within dia according to similarities from pairwise model"""
    cosine_similarities = get_similarities(model, dia)
    
    # add last row because last target never was used as context
    # add first column because first context never was used as target
    cosine_similarities = F.pad(cosine_similarities, pad=(1, 0, 0, 1))
    distance_matrix = 1 - cosine_similarities

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average',
        metric='precomputed'
    ).fit_predict(distance_matrix.cpu().numpy())

    res = [[] for _ in range(len(np.unique(labels)))]
    for i_ut, lab in enumerate(labels):
        res[lab].append(i_ut)
    return res


def make_batch_from_dia(dialogue):
    batch = []
    for i in range(1, len(dialogue)):
        batch.append({
            'context': dialogue[:i],
            'target': dialogue[i]
        })
    return batch


def get_similarities(model, dia):
    batch = make_batch_from_dia(dia)
    context_encodings, target_encodings = model(batch)
    return context_encodings @ target_encodings.T
