from argparse import Namespace
from ...utils.training import train, validate
from ...learners import DialogueEncoderLearner
from ...utils.training import TrainerConfig


def train_or_val(args: Namespace, learner: DialogueEncoderLearner, contrastive_train_loader, val_loaders, trainer_config: TrainerConfig):
    if args.validate:
        validate(
            learner,
            val_loaders,
            trainer_config,
            args,
            'dialogue-encoder'
        )
    else:
        
        train(
            learner,
            contrastive_train_loader,
            val_loaders,
            trainer_config,
            args,
            'dialogue-encoder'
        )
