from ...modeling.dialogue import BaselineDialogueEncoder
from ...utils.training import freeze_hf_model
from ...learners import DialogueEncoderLearnerConfig
from ...modeling.dialogue import BaselineDialogueEncoderConfig


def get_baseline_model(model_config: BaselineDialogueEncoderConfig, learner_config: DialogueEncoderLearnerConfig):
    model = BaselineDialogueEncoder(model_config)
    freeze_hf_model(model.model, learner_config.finetune_layers)

    return model
