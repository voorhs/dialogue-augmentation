if __name__ == "__main__":
    
    # ======= DEFINE TASK =======

    from mylib.utils.training import get_argparser, init_environment
    ap = get_argparser()
    ap.add_argument('--model', dest='model', default='pairwise-cat', choices=[
        'pairwise-cat',
        'pairwise-ema',
        'pairwise-sparse-transformer',
        'pairwise-symmetric-ema'
    ])
    ap.add_argument('--backbone', dest='backbone', default='bert-small', choices=[
        'mpnet',
        'dse-large',
        'dse-base',
        'bert-small'
    ])
    ap.add_argument('--context-size', dest='context_size', default=3, type=int)
    ap.add_argument('--symmetric', dest='symmetric', action='store_true')
    ap.add_argument('--batch-size', dest='batch_size', default=64, type=int)
    ap.add_argument('--topk-metric', dest='k', default=3, type=int)
    ap.add_argument('--num-workers', dest='num_workers', default=2, type=int)
    args = ap.parse_args()

    init_environment(args)

    # ======= DEFINE MODEL =======

    if args.backbone == 'mpnet':
        backbone = 'sentence-transformers/all-mpnet-base-v2'
    elif args.backbone == 'dse-large':
        backbone = 'aws-ai/dse-bert-large'
    elif args.backbone == 'dse-base':
        backbone = 'aws-ai/dse-bert-base'
    elif args.backbone == 'bert-small':
        backbone = 'prajjwal1/bert-small'

    from mylib.utils.training import freeze_hf_model
    from mylib.modeling.aux import mySentenceTransformer
    from mylib.modeling.pairwise import (
        TargetEncoder,
        ContextEncoderConcat,
        ContextEncoderEMA,
        ContextEncoderDM,
        Pairwise
    )
    from mylib.learners import PairwiseLearner, PairwiseLearnerConfig
    from mylib.modeling.dialogue import SparseTransformerDM

    if args.model == 'pairwise-cat':
        learner_config = PairwiseLearnerConfig(
            batch_size=args.batch_size,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=7e-6,
            finetune_layers=3,
            k=args.k,
            temperature=0.05
        )

        encoder = mySentenceTransformer(backbone)
        freeze_hf_model(encoder.model, learner_config.finetune_layers)
        if args.symmetric:
            target_encoder = ContextEncoderConcat(encoder, context_size=args.context_size)
        else:
            target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderConcat(encoder, context_size=args.context_size)
        model = Pairwise(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=256,
            k=learner_config.k,
            temperature=learner_config.temperature,
            hard_negative=False
        )
    elif args.model == 'pairwise-symmetric-concat':
        learner_config = PairwiseLearnerConfig(
            batch_size=args.batch_size,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=7e-6,
            finetune_layers=3,
            k=args.k,
            temperature=0.05
        )

        encoder = mySentenceTransformer(backbone)
        freeze_hf_model(encoder.model, learner_config.finetune_layers)
        target_encoder = ContextEncoderConcat(encoder, context_size=args.context_size)
        context_encoder = ContextEncoderConcat(encoder, context_size=args.context_size)
        model = Pairwise(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=256,
            k=learner_config.k,
            temperature=learner_config.temperature,
            hard_negative=False
        )
    elif args.model == 'pairwise-ema':
        learner_config = PairwiseLearnerConfig(
            batch_size=args.batch_size,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=7e-6,
            finetune_layers=3,
            k=args.k,
            temperature=0.05
        )

        encoder = mySentenceTransformer(backbone)
        freeze_hf_model(encoder.model, learner_config.finetune_layers)
        target_encoder = TargetEncoder(encoder)
        context_encoder = ContextEncoderEMA(encoder, context_size=args.context_size, tau=0.5)
        model = Pairwise(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=256,
            k=learner_config.k,
            temperature=learner_config.temperature,
            hard_negative=False
        )
    elif args.model == 'pairwise-sparse-transformer':
        learner_config = PairwiseLearnerConfig(
            batch_size=args.batch_size,
            # warmup_period=None,
            # do_periodic_warmup=None,
            lr=7e-6,
            finetune_layers=3,
            k=args.k,
            temperature=0.05
        )

        dialogue_model = SparseTransformerDM(backbone)
        freeze_hf_model(dialogue_model.model, learner_config.finetune_layers)

        context_encoder = ContextEncoderDM(dialogue_model, tau=0.5)
        encoder = mySentenceTransformer(backbone, model=dialogue_model.model)
        target_encoder = TargetEncoder(encoder)
        model = Pairwise(
            target_encoder=target_encoder,
            context_encoder=context_encoder,
            projection_size=256,
            context_size=5
        )
    
    # ======= DEFINE LEARNER =======

    if args.weights_from is not None:
        learner = PairwiseLearner.load_from_checkpoint(
            checkpoint_path=args.weights_from,
            model=model,
            config=learner_config
        )
    else:
        learner = PairwiseLearner(model, learner_config)

    # ======= DEFINE DATA =======

    def collate_fn(batch):
        return batch
    
    import os
    root_dir = os.environ['ROOT_DIR']
    path = os.path.join(root_dir, 'data', 'train', 'context-response-pairs')

    from mylib.datasets import ContextResponseDataset
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset=ContextResponseDataset(path, 'train', context_size=args.context_size, symmetric=args.symmetric),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=ContextResponseDataset(path, 'val', context_size=args.context_size, symmetric=args.symmetric),
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )

    # ======= TRAIN =======

    from mylib.utils.training import train

    train(learner, train_loader, val_loader, args, metric_to_monitor='val_metric')
