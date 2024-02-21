if __name__ == "__main__":
    
    # ======= DEFINE TASK =======

    from mylib.utils.training import config_to_argparser, retrieve_fields, TrainerConfig
    from mylib.learners import PairwiseLearner as Learner, PairwiseLearnerConfig as LearnerConfig
    from mylib.modeling.pairwise import Pairwise, PairwiseModelConfig as ModelConfig

    ap = config_to_argparser([ModelConfig, LearnerConfig, TrainerConfig])
    ap.add_argument('--backbone', dest='backbone', default='aws-ai/dse-bert-base')
    ap.add_argument('--data-path', dest='data_path', default='data/source')
    args = ap.parse_args()

    model_config = retrieve_fields(args, ModelConfig)
    learner_config = retrieve_fields(args, LearnerConfig)
    trainer_config = retrieve_fields(args, TrainerConfig)

    from mylib.utils.training import init_environment
    init_environment(args)

    # ======= DEFINE MODEL =======

    from mylib.utils.training import freeze_hf_model
    from mylib.utils.modeling.generic import mySentenceTransformer
    from mylib.modeling.pairwise import Pairwise

    encoder = mySentenceTransformer(args.backbone)
    freeze_hf_model(encoder.model, learner_config.finetune_layers)
    
    model = Pairwise(model=encoder, config=ModelConfig)
    
    # ======= DEFINE LEARNER =======

    if trainer_config.init_from is not None:
        learner = Learner.load_from_checkpoint(
            checkpoint_path=trainer_config.init_from,
            model=model,
            config=learner_config
        )
    else:
        learner = Learner(model, learner_config)

    # ======= DEFINE DATA =======

    def collate_fn(batch):
        return batch

    from mylib.datasets import ContextResponseDataset
    from torch.utils.data import DataLoader

    kwargs = dict(path=args.data_path, context_size=model_config.context_size, symmetric=model_config.symmetric)
    train_dataset = ContextResponseDataset(split='train', **kwargs)
    val_dataset = ContextResponseDataset(split='val', **kwargs)

    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs // learner_config.batch_size
    print('total steps:', learner_config.total_steps)

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

    # ======= TRAIN =======

    from mylib.utils.training import train

    train(learner, train_loader, val_loader, trainer_config, args)
