from argparse import Namespace
from ...utils.training import train, validate
from ...learners import DownstreamClassificationLearner
from ...utils.training import TrainerConfig


def train_or_val(args: Namespace, learner: DownstreamClassificationLearner, train_loader, val_loader, trainer_config: TrainerConfig):
    if args.validate:
        validate(
            learner,
            val_loader,
            trainer_config,
            args,
            'downstream-classification'
        )
    else:
        
        train(
            learner,
            train_loader,
            val_loader,
            trainer_config,
            args,
            'downstream-classification'
        )
