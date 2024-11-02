from active_learning_lab.experiments.argparse import GroupingArgParser, KeywordArgs


DESCRIPTION_DEFAULT = ''
TAGS_DEFAULT = []
RUNS_DEFAULT = 5
ACTIVE_LEARNER_DEFAULT = 'default'
SEED_DEFAULT = 1003
MAX_REPRODUCIBILITY_DEFAULT = False

DATASET_NAME_DEFAULT = ''

CLASSIFIER_NAME_DEFAULT = ''
VALIDATION_SET_SIZE_DEFAULT = 0.1

QUERY_STRATEGY_DEFAULT = 'random'
INITIALIZATION_STRATEGY_DEFAULT = 'random'
STOPPING_CRITERION_DEFAULT = None


def get_parser():
    parser = GroupingArgParser(description='Runs an active learning experiment.')

    parser.add_argument('config_module', default='', help='')

    general_subcmd = parser.add_argument_group('general')
    general_subcmd.add_argument('--description', type=str, default=DESCRIPTION_DEFAULT,
                                help='textual description or comments')
    general_subcmd.add_argument('--tags', type=list, nargs='+', default=TAGS_DEFAULT,
                                help='(mlflow) tags to assign')
    general_subcmd.add_argument('--active_learner', type=str, default=ACTIVE_LEARNER_DEFAULT,
                                help='activer learner type')
    general_subcmd.add_argument('--runs', type=int, default=RUNS_DEFAULT, help='number of repetitions')
    general_subcmd.add_argument('--seed', type=int, default=SEED_DEFAULT, help='random seed')
    general_subcmd.add_argument('--max-reproducibility', action='store_true',
                                default=MAX_REPRODUCIBILITY_DEFAULT, help='maximum reproducibility (slow)')

    ds_group = parser.add_argument_group('dataset')
    ds_group.add_argument('--dataset_name', type=str, help='name of dataset used')
    ds_group.add_argument('--dataset_kwargs', action=KeywordArgs, help='dataset keyword args')

    ds_group = parser.add_argument_group('classifier')
    ds_group.add_argument('--classifier_name', type=str, default=CLASSIFIER_NAME_DEFAULT,
                          help='classifier to use [svm, kimcnn, transformer]')

    ds_group.add_argument('--validation_set_size', type=float,
                          default=VALIDATION_SET_SIZE_DEFAULT, help='size of the validation set '
                                                  '(percentage of the train set)')

    ds_group.add_argument('--classifier_kwargs', action=KeywordArgs, help='classifier keyword args')

    al_group = parser.add_argument_group('active learner')
    al_group.add_argument('--num_queries', type=int, default=10, help='number of active learning queries')
    al_group.add_argument('--query_size', type=str, default=100, help='number of instances returned by a query step (absolute or relative)')

    al_group.add_argument('--active_learner_kwargs', action=KeywordArgs, help='active learner keyword args')

    al_group.add_argument('--initialization_strategy', type=str, default='random', help='initialization to use')
    al_group.add_argument('--initialization_strategy_kwargs', action=KeywordArgs, help='initialization keyword args')

    al_group.add_argument('--query_strategy', type=str, default='random', help='query strategy to use')
    al_group.add_argument('--query_strategy_kwargs', action=KeywordArgs, help='query strategy keyword args')

    al_group.add_argument('--validation_set_sampling', type=str, default='random', help='how to construct the validation set')

    return parser


def get_non_default_args(config_dict):
    args = dict({
        'general': {},
        'dataset': {},
        'classifier': {},
        'active_learner': {}
    })

    if config_dict['general']['description'] != DESCRIPTION_DEFAULT:
        args['general']['description'] = config_dict['general']['description']
    if config_dict['general']['tags'] != TAGS_DEFAULT:
        args['general']['tags'] = config_dict['general']['tags']
    if config_dict['general']['runs'] != RUNS_DEFAULT:
        args['general']['runs'] = config_dict['general']['runs']
    if config_dict['general']['active_learner'] != ACTIVE_LEARNER_DEFAULT:
        args['active_learner']['active_learner_type'] = config_dict['general']['active_learner']
    if config_dict['active_learner']['active_learner_kwargs'] is not None \
            and len(config_dict['active_learner']['active_learner_kwargs']) > 0:
         args['active_learner']['active_learner_kwargs'] = config_dict['active_learner']['active_learner_kwargs']
    else:
        args['active_learner']['active_learner_kwargs'] = dict()
    if config_dict['general']['seed'] != SEED_DEFAULT:
        args['general']['seed'] = config_dict['general']['seed']
    if config_dict['general']['max_reproducibility'] != MAX_REPRODUCIBILITY_DEFAULT:
        args['general']['max_reproducibility'] = config_dict['general']['max_reproducibility']

    if config_dict['dataset']['dataset_name'] != DATASET_NAME_DEFAULT:
        args['dataset']['dataset_name'] = config_dict['dataset']['dataset_name']
    if config_dict['dataset']['dataset_kwargs'] is not None and len(config_dict['dataset']['dataset_kwargs']) > 0:
         args['dataset']['dataset_kwargs'] = config_dict['dataset']['dataset_kwargs']
    else:
        args['dataset']['dataset_kwargs'] = dict()

    if config_dict['classifier']['classifier_name'] != CLASSIFIER_NAME_DEFAULT:
        args['classifier']['classifier_name'] = config_dict['classifier']['classifier_name']
    if config_dict['classifier']['classifier_kwargs'] is not None and len(config_dict['classifier']['classifier_kwargs']) > 0:
         args['classifier']['classifier_kwargs'] = config_dict['classifier']['classifier_kwargs']
    else:
        args['classifier']['classifier_kwargs'] = dict()

    args['active_learner']['query_strategy'] = config_dict['active_learner']['query_strategy']

    if config_dict['active_learner']['query_strategy_kwargs'] is not None and len(config_dict['active_learner']['query_strategy_kwargs']) > 0:
         args['active_learner']['query_strategy_kwargs'] = config_dict['active_learner']['query_strategy_kwargs']
    else:
        args['active_learner']['query_strategy_kwargs'] = dict()

    if config_dict['active_learner']['initialization_strategy'] != INITIALIZATION_STRATEGY_DEFAULT:
        args['active_learner']['initialization_strategy'] = config_dict['active_learner']['initialization_strategy']

    if config_dict['active_learner']['initialization_strategy_kwargs'] is not None and len(config_dict['active_learner']['initialization_strategy_kwargs']) > 0:
         args['active_learner']['initialization_strategy_kwargs'] = config_dict['active_learner']['initialization_strategy_kwargs']
    else:
        args['active_learner']['initialization_strategy_kwargs'] = dict()

    return args
