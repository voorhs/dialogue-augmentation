from ...modeling.dialogue import BaselineDialogueEncoder
from ...utils.training import freeze_hf_model
from ...learners import DownstreamClassificationLearnerConfig, DialogueEncoderLearner, DialogueEncoderLearnerConfig
from ...modeling.dialogue import BaselineDialogueEncoderConfig


def get_baseline_model(model_config: BaselineDialogueEncoderConfig, learner_config: DownstreamClassificationLearnerConfig):
    model = BaselineDialogueEncoder(model_config)
    
    if learner_config.encoder_weights is not None:
        learner = DialogueEncoderLearner.load_from_checkpoint(
            checkpoint_path=learner_config.encoder_weights,
            model=model,
            config=DialogueEncoderLearnerConfig()
        )
        model = learner.model
    
    freeze_hf_model(model.model, learner_config.finetune_layers)

    return model
