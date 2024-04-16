if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('--seed', default=0, type=int)
    ap.add_argument('--path-in', required=True)
    ap.add_argument('--path-out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pooling', choices=['cls', 'avg'], required=True)
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
        truncation=True
    )
    encoder = BaselineDialogueEncoder(config).cuda().eval()

    # load data
    from datasets import load_from_disk, disable_caching
    disable_caching()

    dataset = load_from_disk(args.path_in)
    
    # feed to model and collect embeddings
    from torch import no_grad

    with no_grad():
        dataset = dataset.map(
            lambda batch: {'embedding': encoder(batch).detach().cpu().numpy()},
            input_columns='content',
            batched=True,
            batch_size=args.batch_size
        )

    # save to disk
    dataset.save_to_disk(args.path_out)
