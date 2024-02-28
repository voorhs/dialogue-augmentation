from ...learners import PairwiseLearner, PairwiseLearnerConfig
from ...utils.training import TrainerConfig


def get_learner(model, learner_config: PairwiseLearnerConfig, trainer_config: TrainerConfig):
    if trainer_config.init_from is None:
        return PairwiseLearner(model, learner_config)
    
    return PairwiseLearner.load_from_checkpoint(
        checkpoint_path=trainer_config.init_from,
        model=model,
        config=learner_config
    )
    