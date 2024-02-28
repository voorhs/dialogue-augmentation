from argparse import Namespace
from torch.utils.data import DataLoader
from ...datasets import ContrastiveDataset, DomainDataset, HalvesDataset
from ...learners import DialogueEncoderLearnerConfig
from ...utils.training import TrainerConfig


def get_loaders(args: Namespace, learner_config: DialogueEncoderLearnerConfig, trainer_config: TrainerConfig):
    
    if args.halves:
        contrastive_train = HalvesDataset(args.contrastive_path)
    else:
        contrastive_train = ContrastiveDataset(args.contrastive_path)
    
    
    learner_config.total_steps = len(contrastive_train) * trainer_config.n_epochs // learner_config.batch_size
    # print('total steps:', learner_config.total_steps)

    multiwoz_train = DomainDataset(
        path=args.multiwoz_path,
        split='train'
    )
    multiwoz_val = DomainDataset(
        path=args.multiwoz_path,
        split='validation'
    )
    bitod_train = DomainDataset(
        path=args.bitod_path,
        split='train'
    )
    bitod_val = DomainDataset(
        path=args.bitod_path,
        split='validation'
    )
    sgd_train = DomainDataset(
        path=args.sgd_path,
        split='train'
    )
    sgd_val = DomainDataset(
        path=args.sgd_path,
        split='validation'
    )
    
    def collate_fn(batch):
        return batch
    
    contrastive_train_loader = DataLoader(
        dataset=contrastive_train,
        batch_size=learner_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_args = dict(
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    multiwoz_train_loader = DataLoader(dataset=multiwoz_train, **val_args)
    multiwoz_val_loader = DataLoader(dataset=multiwoz_val, **val_args)

    bitod_train_loader = DataLoader(dataset=bitod_train, **val_args)
    bitod_val_loader = DataLoader(dataset=bitod_val, **val_args)

    sgd_train_loader = DataLoader(dataset=sgd_train, **val_args)
    sgd_val_loader = DataLoader(dataset=sgd_val, **val_args)

    val_loaders = [
        multiwoz_train_loader, multiwoz_val_loader,
        bitod_train_loader, bitod_val_loader,
        sgd_train_loader, sgd_val_loader
    ]

    return contrastive_train_loader, val_loaders
