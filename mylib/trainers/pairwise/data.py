from argparse import Namespace
from torch.utils.data import DataLoader
from ...datasets import ContextResponseDataset
from ...learners import PairwiseLearnerConfig
from ...utils.training import TrainerConfig


def get_loaders(args: Namespace, learner_config: PairwiseLearnerConfig, trainer_config: TrainerConfig):
    def collate_fn(batch):
        return batch

    
    kwargs = dict(path=args.data_path, context_size=args.context_size, symmetric=args.symmetric)
    train_dataset = ContextResponseDataset(split='train', **kwargs)
    val_dataset = ContextResponseDataset(split='val', **kwargs)

    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs // learner_config.batch_size
    # print('total steps:', learner_config.total_steps)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn,
        drop_last=False
    )

    return train_loader, val_loader