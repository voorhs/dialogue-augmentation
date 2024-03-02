from ...utils.training import config_to_argparser, retrieve_fields, TrainerConfig, init_environment
from ...learners import DownstreamClassificationLearnerConfig as LearnerConfig


def get_configs(ModelConfig):
    ap = config_to_argparser([ModelConfig, LearnerConfig, TrainerConfig])
    ap.add_argument('--dataset-path', dest='dataset_path', required=True)
    ap.add_argument('--validate', dest='validate', action='store_true')
    args = ap.parse_args()

    model_config = retrieve_fields(args, ModelConfig)
    learner_config = retrieve_fields(args, LearnerConfig)
    trainer_config = retrieve_fields(args, TrainerConfig)

    init_environment(args)

    return args, model_config, learner_config, trainer_config
