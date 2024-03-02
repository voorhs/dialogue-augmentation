from argparse import Namespace
from torch.utils.data import DataLoader
from ...datasets import DomainDataset
from ...learners import DownstreamClassificationLearnerConfig
from ...utils.training import TrainerConfig


def get_loaders(args: Namespace, learner_config: DownstreamClassificationLearnerConfig, trainer_config: TrainerConfig):
    
    train_dataset = DomainDataset(
        path=args.dataset_path,
        split='train'
    )
    val_dataset = DomainDataset(
        path=args.dataset_path,
        split='validation'
    )
    
    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs // learner_config.batch_size
    
    def collate_fn(batch):
        return batch
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=learner_config.batch_size,
        shuffle=True,
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
