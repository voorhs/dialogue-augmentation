from ..learners import DialogueEncoderLearner, DialogueEncoderLearnerConfig
from ..utils.training import TrainerConfig


def get_learner(model, learner_config: DialogueEncoderLearnerConfig, trainer_config: TrainerConfig):
    if trainer_config.init_from is None:
        return DialogueEncoderLearner(model, learner_config)
    
    return DialogueEncoderLearner.load_from_checkpoint(
        checkpoint_path=trainer_config.init_from,
        model=model,
        config=learner_config
    )
