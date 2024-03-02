from ...learners import BaseLearner
from dataclasses import dataclass
from argparse import Namespace


@dataclass
class TrainerConfig:
    name: str = None
    n_workers: int = 8
    n_epochs: int = 5
    seed: int = 0
    cuda: str = None
    # interval: int = 4000
    logdir: str = './logs'
    logger: str = 'tb'
    resume_from: str = None
    init_from: str = None
    metric_for_checkpoint: str = None
    save_last: bool = True
    save_top_k: int = 1
    mode: str = None


class LightningCkptLoadable:
    """Mixin for `nn.Module`"""
    def load_checkpoint(self, path_to_ckpt, learner_class: BaseLearner, map_location=None):
        model = learner_class.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=self,
            config=learner_class.get_default_config(),
        ).model
        
        self.load_state_dict(model.state_dict())


def freeze_hf_model(hf_model, finetune_layers):
    """Freeze all encoder layers except last `finetune_encoder_layers`"""
    hf_model.requires_grad_(False)
    n_layers = hf_model.config.num_hidden_layers
    for i in range(n_layers):
        hf_model.encoder.layer[i].requires_grad_(i>=n_layers-finetune_layers)


class HParamsPuller:
    def get_hparams(self):
        res = {}
        for attr, val in vars(self).items():
            if hasattr(val, 'get_hparams'):
                tmp = val.get_hparams()
                tmp = self.add_prefix(tmp, attr)
                res.update(tmp)
            elif isinstance(val, (int, float, str, bool)):
                res[attr] = val
        return res
    
    @staticmethod
    def add_prefix(dct, prefix):
        res = {}
        for key, val in dct.items():
            res[f'{prefix}.{key}'] = val
        return res


def train(learner, train_loader, val_loader, config: TrainerConfig, args: Namespace, project_name):
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    import os
    
    checkpoint_callback = ModelCheckpoint(
        # monitor=config.metric_for_checkpoint,
        # save_last=config.save_last,
        # save_top_k=config.save_top_k,
        # mode=config.mode,
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    if config.logger == 'comet':
        import comet_ml
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CometLogger
    import json

    if config.logger == 'tb':
        logger = TensorBoardLogger(
            save_dir=os.path.join('.', 'logs', 'tensorboard'),
            name=config.name
        )
    elif config.logger == 'wb':
        logger = WandbLogger(
            save_dir=os.path.join('.', 'logs', 'wandb'),
            name=config.name
        )
    elif config.logger == 'comet':
        secrets = json.load(open('secrets.json', 'r'))
        logger = CometLogger(
            api_key=secrets['comet_api'],
            workspace=secrets["workspace"],
            save_dir=os.path.join('.', 'logs', 'comet'),
            project_name=project_name,
            experiment_name=config.name,
        )
    else:
        raise ValueError('unknown logger name')

    logger.log_hyperparams(vars(args))

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        # max_time={'hours': 24},
        
        # max_time={'minutes': 2},
        # max_steps=0,

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",
        devices=-1,
        strategy='ddp',

        # fraction of data to use
        limit_train_batches=1.,
        limit_val_batches=0,

        # logging and checkpointing
        # val_check_interval=config.interval,   # number of optimization steps between two validation runs
        check_val_every_n_epoch=1,
        logger=logger,
        enable_progress_bar=False,
        profiler=None,
        callbacks=callbacks,
        # log_every_n_steps=5,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    # if args.resume_from is None:
    #     trainer.validate(learner, val_loader)

    # from datetime import datetime
    # print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.fit(
        learner, train_loader, val_loader,
        ckpt_path=config.resume_from
    )

    # print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # trainer.validate(learner, val_loader, ckpt_path='best')


def validate(learner, val_loader, config: TrainerConfig, args: Namespace, project_name):
    from lightning.pytorch.callbacks import LearningRateMonitor
    import os
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]

    if config.logger == 'comet':
        import comet_ml
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CometLogger
    import json

    if config.logger == 'tb':
        logger = TensorBoardLogger(
            save_dir=os.path.join('.', 'logs', 'tensorboard'),
            name=config.name
        )
    elif config.logger == 'wb':
        logger = WandbLogger(
            save_dir=os.path.join('.', 'logs', 'wandb'),
            name=config.name
        )
    elif config.logger == 'comet':
        secrets = json.load(open('secrets.json', 'r'))
        logger = CometLogger(
            api_key=secrets['comet_api'],
            workspace=secrets["workspace"],
            save_dir=os.path.join('.', 'logs', 'comet'),
            project_name=project_name,
            experiment_name=config.name,
        )
    else:
        raise ValueError('unknown logger name')

    logger.log_hyperparams(vars(args))

    trainer = pl.Trainer(
        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",
        # devices=-1,
        # strategy='ddp',

        # fraction of data to use
        limit_val_batches=1.,

        # logging and checkpointing
        logger=logger,
        enable_progress_bar=False,
        profiler=None,
        callbacks=callbacks,
    )

    if config.init_from is None:
        raise ValueError('provide path to checkpoint via `--init-from PATH` argument')

    trainer.validate(learner, val_loader, ckpt_path=config.init_from)


def init_environment(args):
    import torch
    torch.set_float32_matmul_precision('medium')

    seed_everything(args.seed)

    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


def seed_everything(seed: int):
    """https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964"""
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def config_to_argparser(container_classes):
    from dataclasses import fields
    def add_arguments(container_class, parser):
        for field in fields(container_class):
            parser.add_argument(
                '--' + field.name.replace('_', '-'),
                dest=field.name,
                default=field.default,
                type=field.type
            )
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    for cls in container_classes:
        add_arguments(cls, parser)

    return parser


def retrieve_fields(namespace, contrainer_class):
    from dataclasses import fields
    res = {}
    for field in fields(contrainer_class):
        res[field.name] = getattr(namespace, field.name)
    return contrainer_class(**res)
