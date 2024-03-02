from ...learners import DownstreamClassificationLearner, DownstreamClassificationLearnerConfig
from ...utils.training import TrainerConfig


def get_learner(model, learner_config: DownstreamClassificationLearnerConfig, trainer_config: TrainerConfig):
    if trainer_config.init_from is None:
        return DownstreamClassificationLearner(model, learner_config)
    
    return DownstreamClassificationLearner.load_from_checkpoint(
        checkpoint_path=trainer_config.init_from,
        model=model,
        config=learner_config
    )
