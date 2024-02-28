from ...utils.training import config_to_argparser, retrieve_fields, TrainerConfig, init_environment
from ...learners import DialogueEncoderLearnerConfig as LearnerConfig


def get_configs(ModelConfig):
    ap = config_to_argparser([ModelConfig, LearnerConfig, TrainerConfig])
    ap.add_argument('--contrastive-path', dest='contrastive_path', required=True)
    ap.add_argument('--multiwoz-path', dest='multiwoz_path', required=True)
    ap.add_argument('--bitod-path', dest='bitod_path', required=True)
    ap.add_argument('--sgd-path', dest='sgd_path', required=True)
    ap.add_argument('--validate', dest='validate', action='store_true')
    ap.add_argument('--halves', dest='halves', action='store_true')
    args = ap.parse_args()

    model_config = retrieve_fields(args, ModelConfig)
    learner_config = retrieve_fields(args, LearnerConfig)
    trainer_config = retrieve_fields(args, TrainerConfig)

    init_environment(args)

    return args, model_config, learner_config, trainer_config
