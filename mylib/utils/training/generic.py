from ...learners import BaseLearner


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


def train(learner, train_loader, val_loader, args, metric_to_monitor):
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    import os

    if args.logger != 'none':
        checkpoint_callback = ModelCheckpoint(
            monitor=metric_to_monitor,
            save_last=True,
            save_top_k=1,
            mode='max',
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback, lr_monitor]
    else:
        callbacks = None

    import lightning.pytorch as pl
    if args.logger == 'tb':
        Logger = pl.loggers.TensorBoardLogger
        suffix = 'tensorboard'
    elif args.logger == 'wb':
        Logger = pl.loggers.WandbLogger
        suffix = 'wandb'
    elif args.logger == 'none':
        Logger = lambda **kwargs: False
        suffix = ''
    
    logger = Logger(
        save_dir=os.path.join('.', 'logs', suffix),
        name=args.name
    )

    trainer = pl.Trainer(
        # max_epochs=1,
        max_time={'hours': 14},
        
        # max_time={'minutes': 10},
        # max_steps=0,

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        precision="16-mixed",

        # logging and checkpointing
        val_check_interval=args.interval,
        # check_val_every_n_epoch=1,
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

    if args.resume_from is None:
        trainer.validate(learner, val_loader)

    from datetime import datetime
    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.fit(
        learner, train_loader, val_loader,
        ckpt_path=args.resume_from
    )

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.validate(learner, val_loader)


def get_argparser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', dest='name', default=None)
    ap.add_argument('--cuda', dest='cuda', default='0')
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    ap.add_argument('--interval', dest='interval', default=500, type=int)
    ap.add_argument('--logger', dest='logger', choices=['none', 'tb', 'wb'], default='tb')
    ap.add_argument('--resume-training-from', dest='resume_from', default=None)
    ap.add_argument('--load-weights-from', dest='weights_from', default=None)
    return ap


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
