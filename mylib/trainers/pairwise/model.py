from ...utils.training import freeze_hf_model
from ...utils.modeling.generic import mySentenceTransformer
from ...modeling.pairwise import Pairwise, PairwiseModelConfig
from ...learners import PairwiseLearnerConfig


def get_model(model_config: PairwiseModelConfig, learner_config: PairwiseLearnerConfig):
    encoder = mySentenceTransformer(model_config.hf_model)
    freeze_hf_model(encoder.model, learner_config.finetune_layers)
    
    return Pairwise(model=encoder, config=model_config)
