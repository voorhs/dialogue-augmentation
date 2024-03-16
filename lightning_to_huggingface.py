from glob import glob
from transformers import AutoTokenizer
from mylib.modeling.dialogue import BaselineDialogueEncoderConfig, BaselineDialogueEncoder
from mylib.learners import DialogueEncoderLearner, DialogueEncoderLearnerConfig


checkpoint_codenames ={
    'google-bert/bert-base-uncased':
    {
        'trivial': "ba6388e710684d94af81ebab88c2ff5a",
        'advanced': "0d0b24e2d3864cf8a6c8b7d5e15b1c83",
        'crazy' :"d7e24b7b434e486d866f35d70ac0c503",
        'halves': "dd2c00a94bab43d5a85588782d1c16a9",
        'halves-dropout': "e1d6e434bef840b78509477dbc8505d3",
    },
    'FacebookAI/roberta-base':
    {
        'trivial': "e4fa2d98923e4c8ca4fbe20c7ccc4c67",
        'advanced': "849fa15859484683a4341837e7d1dd1d",
        'crazy': "523da04c3c5a41379acc298ee09c7a0a",
        'halves': "caee9e3358d9476fa0f6d1504740239e",
        'halves-dropout': "1509ea32ae0a4fe29a0048a57954fcfc",
    },
    'Shitao/RetroMAE':
    {
        'trivial': "408af9ac5d114a9da1fa007277343f15",
        'advanced': "e2de548066f14753bdf9a646e992c02f",
        'crazy' :"ca66a8cd07c4418daea805093d00cdd3",
        'halves': "e74b5f8cfd5849ee88715b784f731b3c",
        'halves-dropout': "30b0491f250845eb89223c9abb4ef5bb",
    }
}


def get_raw_model(model_config: BaselineDialogueEncoderConfig):
    model = BaselineDialogueEncoder(model_config)
    model.requires_grad_(False)
    return model


def get_pretrained_model(model, init_from):
    return DialogueEncoderLearner.load_from_checkpoint(
        checkpoint_path=init_from,
        model=model,
        config=DialogueEncoderLearnerConfig(),
        map_location='cpu'
    ).model.model


def get_files(ckpt_code):
    return glob(f"logs/comet/dialogue-encoder/{ckpt_code}/checkpoints/*.ckpt")


def to_hugging_face(ckpt_path, model_config, path_out):
    model = get_raw_model(model_config)
    model = get_pretrained_model(model, ckpt_path)
    model.save_pretrained(path_out)
    tok = AutoTokenizer.from_pretrained(model_config.hf_model)
    tok.save_pretrained(path_out)


if __name__ == "__main__":
    for backbone_name in checkpoint_codenames.keys():
        model_config = BaselineDialogueEncoderConfig(hf_model=backbone_name)
        for aug_set in checkpoint_codenames[backbone_name].keys():
            for ckpt in get_files(checkpoint_codenames[backbone_name][aug_set]):
                backbone_for_path = backbone_name.replace('/', '-')
                ckpt_for_path = ckpt.split('/')[-1].split('-')[0]
                to_hugging_face(ckpt, model_config, f'pretrained/{backbone_for_path}/{aug_set}/{ckpt_for_path}')
