if __name__ == "__main__":

    from mylib.utils.training import config_to_argparser, retrieve_fields, TrainerConfig
    from mylib.learners import DialogueEncoderLearner, DialogueEncoderLearnerConfig as LearnerConfig
    from mylib.modeling.dialogue.baseline_dialogue_encoder import BaselineDialogueEncoderConfig as ModelConfig
    
    ap = config_to_argparser([ModelConfig, LearnerConfig, TrainerConfig])
    ap.add_argument('--contrastive-path', dest='contrastive_path', required=True)
    ap.add_argument('--multiwoz-path', dest='multiwoz_path', required=True)
    args = ap.parse_args()

    model_config = retrieve_fields(args, ModelConfig)
    learner_config = retrieve_fields(args, LearnerConfig)
    trainer_config = retrieve_fields(args, TrainerConfig)

    from mylib.utils.training import init_environment
    init_environment(args)

    # ======= DEFINE DATA =======

    from mylib.datasets import ContrastiveDataset, MultiWOZServiceClfDataset
    contrastive_train = ContrastiveDataset(args.contrastive_path)
    
    multiwoz_train = MultiWOZServiceClfDataset(
        path=args.multiwoz_path,
        split='train'
    )
    multiwoz_val = MultiWOZServiceClfDataset(
        path=args.multiwoz_path,
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
    multiwoz_train_loader = DataLoader(
        dataset=multiwoz_train,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    multiwoz_val_loader = DataLoader(
        dataset=multiwoz_val,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn
    )

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

    from mylib.utils.training import train
    train(
        learner,
        contrastive_train_loader,
        [multiwoz_train_loader, multiwoz_val_loader],
        trainer_config,
        args
    )
