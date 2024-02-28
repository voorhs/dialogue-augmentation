from ...modeling.dialogue import HSSADialogueEncoder, HSSADialogueEncoderConfig
from ...modeling.hssa.utils import freeze_hssa
from ...learners import DialogueEncoderLearnerConfig


def get_hssa_model(model_config: HSSADialogueEncoderConfig, learner_config: DialogueEncoderLearnerConfig):
    model = HSSADialogueEncoder(model_config)
    freeze_hssa(model.model, learner_config.finetune_layers)

    return model
