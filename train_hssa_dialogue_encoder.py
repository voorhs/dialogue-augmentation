if __name__ == "__main__":
    from mylib.modeling.dialogue import HSSADialogueEncoderConfig
    from mylib.trainers.encoder import get_configs, get_loaders, get_hssa_model, get_learner, train_or_val

    args, model_config, learner_config, trainer_config = get_configs(HSSADialogueEncoderConfig)
    contrastive_train_loader, val_loaders = get_loaders(args, learner_config, trainer_config)
    model = get_hssa_model(model_config, learner_config)
    learner = get_learner(model, learner_config, trainer_config)
    train_or_val(args, learner, contrastive_train_loader, val_loaders, trainer_config)