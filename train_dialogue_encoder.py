if __name__ == "__main__":

    from mylib.utils.training import config_to_argparser, retrieve_fields, TrainerConfig
    from mylib.learners import DialogueEncoderLearner, DialogueEncoderLearnerConfig as LearnerConfig
    from mylib.modeling.dialogue.baseline_dialogue_encoder import BaselineDialogueEncoderConfig as ModelConfig
    
    ap = config_to_argparser([ModelConfig, LearnerConfig, TrainerConfig])
    ap.add_argument('--contrastive-path', dest='contrastive_path', required=True)
    ap.add_argument('--multiwoz-path', dest='multiwoz_path', required=True)
    ap.add_argument('--bitod-path', dest='bitod_path', required=True)
    ap.add_argument('--sgd-path', dest='sgd_path', required=True)
    ap.add_argument('--validate', dest='validate', action='store_true')
    args = ap.parse_args()

    model_config = retrieve_fields(args, ModelConfig)
    learner_config = retrieve_fields(args, LearnerConfig)
    trainer_config = retrieve_fields(args, TrainerConfig)

    from mylib.utils.training import init_environment
    init_environment(args)

    # ======= DEFINE DATA =======

    from mylib.datasets import ContrastiveDataset, DomainDataset
    contrastive_train = ContrastiveDataset(args.contrastive_path)
    
    learner_config.total_steps = len(contrastive_train) * trainer_config.n_epochs // learner_config.batch_size
    print('total steps:', learner_config.total_steps)

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

    from torch.utils.data import DataLoader
    
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

    # ======= DEFINE MODEL =======

    from mylib.modeling.dialogue import BaselineDialogueEncoder
    from mylib.utils.training import freeze_hf_model
    
    model = BaselineDialogueEncoder(model_config)
    freeze_hf_model(model.model, learner_config.finetune_layers)

    # ======= DEFINE LEARNER =======

    if trainer_config.init_from is not None:
        learner = DialogueEncoderLearner.load_from_checkpoint(
            checkpoint_path=trainer_config.init_from,
            model=model,
            config=learner_config
        )
    else:
        learner = DialogueEncoderLearner(model, learner_config)

    # ======= DEFINE TRAINER =======

    if args.validate:
        from mylib.utils.training import validate
        validate(
            learner,
            val_loaders,
            trainer_config,
            args,
            'dialogue-encoder'
        )
    else:
        from mylib.utils.training import train
        train(
            learner,
            contrastive_train_loader,
            val_loaders,
            trainer_config,
            args,
            'dialogue-encoder'
        )
