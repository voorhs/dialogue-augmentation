if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('--seed', default=0, type=int)
    ap.add_argument('--path-in', required=True)
    ap.add_argument('--path-out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pooling', choices=['cls', 'avg', 'last'], required=True)
    ap.add_argument('--batch-size', required=True, type=int)
    ap.add_argument('--cuda', required=True)
    args = ap.parse_args()

    from mylib.utils.training import init_environment
    init_environment(args)

    # load model
    from mylib.modeling.dialogue import BaselineDialogueEncoder, BaselineDialogueEncoderConfig
    
    config = BaselineDialogueEncoderConfig(
        hf_model=args.model,
        pooling=args.pooling,
        truncation=True,
        max_length=4096 if args.pooling == 'last' else 512
    )

    encoder = BaselineDialogueEncoder(config).cuda().eval()

    # load data
    from datasets import load_from_disk, disable_caching
    disable_caching()

    dataset = load_from_disk(args.path_in)
    
    # feed to model and collect embeddings
    from mylib.utils.modeling import AveragePooling

    pooler = AveragePooling().to(encoder.device)

    def get_hidden_states(batch):
        embeddings, hidden_states = encoder.get_all_hidden_states(batch, pooler)
        return dict(
            embedding=embeddings.detach().cpu().numpy(),
            hidden_states=hidden_states.detach().cpu().numpy()
        )

    from torch import no_grad

    with no_grad():
        dataset = dataset.map(
            get_hidden_states,
            fn_kwargs=dict(pooler=pooler),
            input_columns='content',
            batched=True,
            batch_size=args.batch_size
        )

    # save to disk
    dataset.save_to_disk(args.path_out)
