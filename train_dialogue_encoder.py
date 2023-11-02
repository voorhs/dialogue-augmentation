if __name__ == "__main__":
    
    from mylib.utils.training import get_argparser, init_environment
    ap = get_argparser()
    ap.add_argument('--data-path', dest='data_path', required=True)
    ap.add_argument('--hf-model', dest='hf_model', required=True)
    args = ap.parse_args()

    init_environment(args)

    from mylib.modeling.dialogue import BaselineDialogueEncoder
    from mylib.learners import DialogueEcoderLearner, DialogueEcoderLearnerConfig
    from mylib.utils.training import freeze_hf_model
    
    learner_config = DialogueEcoderLearnerConfig(
        k=1,
        temperature=0.1,
        batch_size=32,
        # warmup_period=200,
        do_periodic_warmup=False,
        lr=3e-6,
        finetune_layers=1
    )
    
    model = BaselineDialogueEncoder(args.hf_model)
    freeze_hf_model(model.model, learner_config.finetune_layers)

    # ======= DEFINE LEARNER =======

    if args.weights_from is not None:
        learner = DialogueEcoderLearner.load_from_checkpoint(
            checkpoint_path=args.weights_from,
            model=model,
            config=learner_config
        )
    else:
        learner = DialogueEcoderLearner(model, learner_config)

    learner = DialogueEcoderLearner(model, learner_config)

    # ======= DEFINE DATA =======

    import os
    # root_dir = os.environ['ROOT_DIR']
    contrastive_path = os.path.join(args.data_path, 'contrastive')
    multiwoz_path = os.path.join(args.data_path, 'multiwoz22')

    from mylib.datasets import ContrastiveDataset, MultiWOZServiceClfDataset
    contrastive_train = ContrastiveDataset(os.path.join(contrastive_path, 'train'))
    # contrastive_val = ContrastiveDataset(os.path.join(contrastive_path, 'val'))
    
    multiwoz_train = MultiWOZServiceClfDataset(
        path=os.path.join(multiwoz_path, 'train'),
        fraction=learner_config.multiwoz_train_frac
    )
    multiwoz_val = MultiWOZServiceClfDataset(
        path=os.path.join(multiwoz_path, 'validation'),
        fraction=learner_config.multiwoz_val_frac
    )

    from torch.utils.data import DataLoader
    def collate_fn(batch):
        return batch
    contrastive_train_loader = DataLoader(
        dataset=contrastive_train,
        batch_size=learner_config.batch_size,
        shuffle=True,
        num_workers=3,
        collate_fn=collate_fn,
        drop_last=True
    )
    # contrastive_val_loader = DataLoader(
    #     dataset=contrastive_val,
    #     batch_size=learner_config.batch_size,
    #     shuffle=False,
    #     num_workers=3,
    #     collate_fn=collate_fn
    # )
    multiwoz_train_loader = DataLoader(
        dataset=multiwoz_train,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn,
        drop_last=False
    )
    multiwoz_val_loader = DataLoader(
        dataset=multiwoz_val,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=3,
        collate_fn=collate_fn
    )

    # ======= DEFINE TRAINER =======

    from mylib.utils.training import train

    train(learner, contrastive_train_loader, [multiwoz_train_loader, multiwoz_val_loader], args, metric_to_monitor='ranking_metric')
