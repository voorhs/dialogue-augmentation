if __name__ == "__main__":
    
    # ======= DEFINE TASK =======

    import torch
    torch.set_float32_matmul_precision('medium')    

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', dest='model', required=True, choices=[
        'pairwise-cat', 'pairwise-ema', 'pairwise-sparse-transformer',
        'listwise-utterance-transformer', 'listwise-sparse-transformer',
        'listwise-hssa'
    ])
    ap.add_argument('--name', dest='name', default=None)
    ap.add_argument('--cuda', dest='cuda', default='0')
    ap.add_argument('--interval', dest='interval', required=True, type=int)
    args = ap.parse_args()

    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # ======= DEFINE LEARNER =======

    from models.hssa.modeling_hssa import SegmentPooler
    def freeze_hssa(model, finetune_layers=0):
        model.embeddings.requires_grad_(False)
        model.embeddings.word_embeddings.weight[-2:].requires_grad_(True)

        model.encoder.requires_grad_(False)
        for i, layer in enumerate(model.encoder.layer):
            layer.requires_grad_(i>=model.config.num_hidden_layers-finetune_layers)

        for module in model.modules():
            if isinstance(module, SegmentPooler):
                module.requires_grad_(True)

    def freeze_hf_model(hf_model, finetune_layers):
        """Freeze all encoder layers except last `finetune_encoder_layers`"""
        hf_model.requires_grad_(False)
        n_layers = hf_model.config.num_hidden_layers
        for i in range(n_layers):
            hf_model.encoder.layer[i].requires_grad_(i>=n_layers-finetune_layers)

    mpnet_name = 'sentence-transformers/all-mpnet-base-v2'
    amazon_name = 'aws-ai/dse-roberta-base'

    from models.train_utils import Learner, LearnerConfig

    from models.aux import mySentenceTransformer
    from models.pairwise import TargetEncoder, ContextEncoderConcat, ContextEncoderEMA, ContextEncoderDM, ChainCosine
    from models.dialogue import UtteranceTransformerDM, UtteranceTransformerDMConfig, SparseTransformerDM, HSSAConfig, HSSADM
    from models.listwise import UtteranceSorter

    if args.model == 'pairwise-cat':
        context_size = 3
        finetune_encoder_layers = 0
        encoder_name = amazon_name
        k = 5
        temperature = 0.05
        hard_negative = False

        encoder = mySentenceTransformer(encoder_name)
        freeze_hf_model(encoder.model, finetune_encoder_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderConcat(encoder, context_size=context_size)
        model = ChainCosine(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=512,
            context_size=context_size,
            k=k,
            temperature=temperature,
            hard_negative=hard_negative
        )
        learner_config = LearnerConfig(
            batch_size=128,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=3e-5,
            kwargs={
                'context_size': context_size,
                'finetune_encoder_layers': finetune_encoder_layers,
                'encoder_name': encoder_name,
                'k': k,
                'temperature': temperature,
                'hard_negative': hard_negative
            }
        )
    elif args.model == 'pairwise-ema':
        context_size = 3
        finetune_encoder_layers = 0
        tau = 0.5
        encoder_name = amazon_name
        k = 5
        temperature = 0.05
        hard_negative = False

        encoder = mySentenceTransformer(encoder_name)
        freeze_hf_model(encoder.model, finetune_encoder_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderEMA(encoder, context_size=context_size, tau=tau)
        model = ChainCosine(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=512,
            context_size=context_size,
            k=k,
            temperature=temperature,
            hard_negative=hard_negative
        )
        learner_config = LearnerConfig(
            batch_size=96,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=3e-5,
            kwargs={
                'context_size': context_size,
                'finetune_encoder_layers': finetune_encoder_layers,
                'tau': tau,
                'encoder_name': encoder_name,
                'k': k,
                'temperature': temperature,
                'hard_negative': hard_negative
            }
        )
    elif args.model == 'pairwise-sparse-transformer':
        context_size = 6
        finetune_layers = 2
        tau = 0.5
        encoder_name = amazon_name

        dialogue_model = SparseTransformerDM(encoder_name)
        freeze_hf_model(dialogue_model.model, finetune_layers)

        context_encoder = ContextEncoderDM(dialogue_model, tau=tau)
        encoder = mySentenceTransformer(encoder_name, model=dialogue_model.model)
        target_encoder = TargetEncoder(encoder)
        model = ChainCosine(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=512,
            context_size=context_size
        )
        learner_config = LearnerConfig(
            batch_size=128,
            warmup_period=None,
            do_periodic_warmup=False,
            lr=3e-6,
            kwargs={
                'context_size': context_size,
                'finetune_layers': finetune_layers,
                'tau': tau
            }
        )
    elif args.model == 'listwise-utterance-transformer':
        head_dropout_prob = 0.02
        finetune_encoder_layers = 1
        encoder_name = amazon_name
        config = UtteranceTransformerDMConfig(
            num_attention_heads=4,
            attention_probs_dropout_prob=0.02,
            n_layers=4,
            encoder_name=encoder_name,
            embed_turn_ids=False,
            is_casual=False
        )
        dialogue_model = UtteranceTransformerDM(config)
        freeze_hf_model(dialogue_model.encoder.model, finetune_encoder_layers)
        
        model = UtteranceSorter(
            dialogue_model=dialogue_model,
            dropout_prob=head_dropout_prob
        )
        learner_config = LearnerConfig(
            batch_size=192,
            warmup_period=200,
            do_periodic_warmup=False,
            lr=3e-6,
            kwargs={
                'dropout_prob': head_dropout_prob,
                'finetune_encoder_layers': finetune_encoder_layers,
                'encoder_name': encoder_name
            }
        )
    elif args.model == 'listwise-sparse-transformer':
        head_dropout_prob = 0.02
        finetune_layers = 2
        dialogue_model = SparseTransformerDM(mpnet_name)
        freeze_hf_model(dialogue_model.model, finetune_layers)
        
        model = UtteranceSorter(
            dialogue_model=dialogue_model,
            dropout_prob=head_dropout_prob
        )
        learner_config = LearnerConfig(
            batch_size=32,
            warmup_period=200,
            do_periodic_warmup=False,
            lr=1e-5,
            kwargs={
                'dropout_prob': head_dropout_prob,
                'finetune_layers': finetune_layers
            }
        )
    elif args.model == 'listwise-hssa':
        # doesn't work for some reason
        head_dropout_prob = 0.02
        finetune_layers = 1
        config = HSSAConfig(
            max_turn_embeddings=20,
            casual_utterance_attention=False,
            pool_utterances=True
        )
        dialogue_model = HSSADM(mpnet_name, config)
        freeze_hssa(dialogue_model.model, 2)
        model = UtteranceSorter(
            dialogue_model=dialogue_model,
            dropout_prob=head_dropout_prob
        )
        learner_config = LearnerConfig(
            batch_size=32,
            warmup_period=200,
            do_periodic_warmup=True,
            lr=3e-6,
            kwargs={
                'dropout_prob': head_dropout_prob,
                'finetune_layers': finetune_layers
            }
        )

    # learner = Learner.load_from_checkpoint(
    #     checkpoint_path='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training/pairwise-ema-amazon-resumed/checkpoints/last.ckpt',
    #     model=model,
    #     config=learner_config
    # )
    learner = Learner(model, learner_config)

    # ======= DEFINE DATA =======

    from models.train_utils import NUPDataset, DialogueDataset
    dataset = NUPDataset if args.model.startswith('pairwise') else DialogueDataset

    from torch.utils.data import DataLoader
    def collate_fn(batch):
        return batch
    train_loader = DataLoader(
        dataset=dataset('.', 'train', fraction=1.),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=dataset('.', 'val', fraction=.2),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    # ======= DEFINE TRAINER =======

    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric',
        save_last=True,
        save_top_k=3,
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    import lightning.pytorch as pl
    logger = pl.loggers.TensorBoardLogger(
        save_dir='.',
        version=args.name,
        name='/home/alekseev_ilya/dialogue-augmentation/nup/logs/training'
    )

    trainer = pl.Trainer(
        # max_epochs=1,
        max_time={'hours': 24},
        
        # max_time={'minutes': 5},
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
        callbacks=[checkpoint_callback, lr_monitor],
        # log_every_n_steps=5,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    # ======= START TRAINING =======

    from datetime import datetime
    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # do magic!
    trainer.fit(
        learner, train_loader, val_loader
    )

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))