from glob import glob
from transformers import AutoTokenizer
from mylib.modeling.dialogue import BaselineDialogueEncoderConfig, BaselineDialogueEncoder
from mylib.learners import DialogueEncoderLearner, DialogueEncoderLearnerConfig


checkpoint_codenames ={
    'google-bert/bert-base-uncased':
    {
        # 'trivial': "ba6388e710684d94af81ebab88c2ff5a",
        # 'advanced': "0d0b24e2d3864cf8a6c8b7d5e15b1c83",
        # 'crazy' :"d7e24b7b434e486d866f35d70ac0c503",
        # 'halves': "dd2c00a94bab43d5a85588782d1c16a9",
        # 'halves-dropout': "e1d6e434bef840b78509477dbc8505d3",
        # 'trivial-light': "d04c1de3f68440899e4463c6b467bde1",
        # 'trivial-light-mixed': "8d19505a49b842ba8b2b72717eb0c2ff",
        # 'advanced-mixed': "0e19c7c3f39c4f238bb6293b771d17fd",
        # 'trivial-heavy-mixed': "b19de997b67b4c21baa61ce8fe4722a9",
        # 'advanced-light-mixed': "6d6dcd8d53c343b2b2ae88dc140af753",
        # 'advanced-light-dse-mixed': "8ed0e90f07404a87b6b13ed37d67b0e3",
    },
    'FacebookAI/roberta-base':
    {
        # 'trivial': "e4fa2d98923e4c8ca4fbe20c7ccc4c67",
        # 'advanced': "849fa15859484683a4341837e7d1dd1d",
        # 'crazy': "523da04c3c5a41379acc298ee09c7a0a",
        # 'halves': "caee9e3358d9476fa0f6d1504740239e",
        # 'halves-dropout': "1509ea32ae0a4fe29a0048a57954fcfc",
        # 'trivial-light': "8dc593b1b1a44946a004be496c5fb242",
        # 'trivial-light-mixed': "5bf2e076fe5c46f29b6dba6839670cc8",
        # 'advanced-mixed': "335b5076295c408fbb02a67cc6a09705",
        # 'trivial-heavy-mixed': "27080f80029349e4a91d506060f2584b",
        # 'advanced-light-mixed': "88866b87989c49fbb8e01cd5b7d5ae60",
        # 'advanced-light-dse-mixed': "ab2d2970e5f240ddbe050e1fdb589d29"
        
    },
    'Shitao/RetroMAE':
    {
        # 'trivial': "408af9ac5d114a9da1fa007277343f15",
        # 'advanced': "e2de548066f14753bdf9a646e992c02f",
        # 'crazy' :"ca66a8cd07c4418daea805093d00cdd3",
        # 'halves': "e74b5f8cfd5849ee88715b784f731b3c",
        # 'halves-dropout': "30b0491f250845eb89223c9abb4ef5bb",
        # 'trivial-light': "ac2cd44f7c13474e9d68dee28f3ba2c8",
        # 'trivial-light-mixed': "fa671f2d23bd4b26a0dd3230d6a5b39b",
        # 'advanced-mixed': "622b1b8534a54f4fa400f1a5264b62e1",
        # 'trivial-heavy-mixed': "4df5a5164cfb478692a7a6ca6ebf153b",
        # 'advanced-light-mixed': "b5726bb95a7844c1a5bc54bc1678fd66",
        'advanced-light-dse-mixed': "48e93c08f6484d0fb19d75a952e416b3"
    },
    'aws-ai/dse-bert-base':
    {
        # 'advanced-light-dse-mixed': "e82a8460c4c247958b0c5ec5a0bcfcf9",
        'advanced-light-dse-mixed': "419b505257e3447d8219272ea83c6d59"
    },
    'BAAI/bge-base-en-v1.5':
    {
        'advanced-light-dse-mixed': '3b02e4141f7f42c3946f571eeb34868c'
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
