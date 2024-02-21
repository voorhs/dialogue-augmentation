from dataclasses import dataclass
from .generic import BaseLearner, BaseLearnerConfig
from .loss import contrastive_loss
from .accuracy import all_accuracies


@dataclass
class PairwiseLearnerConfig(BaseLearnerConfig):
    finetune_layers: int = 3
    temperature: float = 0.05
    loss: str = 'contrastive_cross'   # 'contrastive_cross', 'contrastive_symmetric', 'contrastive_bce'


class PairwiseLearner(BaseLearner):
    def __init__(self, model, config: PairwiseLearnerConfig):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, batch):
        context_encodings, target_encodings = self.model(batch)
        pairwise_scores = context_encodings @ target_encodings.T / self.config.temperature

        loss = contrastive_loss(pairwise_scores, self.config.loss)
        metric = all_accuracies(pairwise_scores)
        
        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric_ = self.forward(batch)

        metric = {}
        for key, val in metric_.items():
            metric[f'train_{key}'] = val

        metric['train_loss'] = loss        
  
        self.log_dict(
            dictionary=metric,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.config.batch_size
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, metric_ = self.forward(batch)
        
        metric = {}
        for key, val in metric_.items():
            metric[f'val_{key}'] = val

        metric['val_loss'] = loss        

        self.log_dict(
            dictionary=metric,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.config.batch_size
        )

    @staticmethod
    def get_default_config():
        return PairwiseLearnerConfig()
