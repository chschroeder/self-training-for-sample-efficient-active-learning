import json

import contextlib

import mlflow
import torch

from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from types import ModuleType

from dependency_injector import containers, providers
from dependency_injector.providers import Configuration
from dependency_injector.wiring import inject, Provide
from mergedeep import merge, Strategy
from torch.utils.tensorboard import SummaryWriter

from active_learning_lab.data.dataset_loaders import DatasetLoader
from active_learning_lab.experiments.active_learning.active_learning_args import get_parser, get_non_default_args
from active_learning_lab.experiments.active_learning.active_learning_experiment_builder import get_active_learner_builder
from active_learning_lab.experiments.tracking import log_mlflow_artifacts, track_environment_info
from active_learning_lab.utils.experiment import set_random_seed
from active_learning_lab.utils.logging import setup_logger, log_args, log_experiment_info, StreamToLogger

from active_learning_lab.utils.mlflow import get_experiment, get_mlflow_tmp_path


class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    mlflow_run = providers.Singleton(
        mlflow.start_run,
        experiment_id=config.experiment.id.as_(str)
    )

    dataset_loader = providers.Singleton(
        DatasetLoader
    )

    summary_writer = providers.Singleton(SummaryWriter)


@inject
def main(config_module: ModuleType,
         base_args: dict,
         override_args: dict,
         config: dict = Provide[Container.config],
         mlflow_run: mlflow.ActiveRun = Provide[Container.mlflow_run]) -> None:

    tmp_path = get_mlflow_tmp_path(mlflow_run)
    log_file_path = tmp_path.joinpath('out.log').resolve()

    try:
        with TemporaryDirectory(dir=str(tmp_path.resolve())) as tmp_dir:

            logger = setup_logger(log_file_path)
            logger.info(f'output log temp: {log_file_path}')

            stream = StreamToLogger(logger)
            with contextlib.redirect_stderr(stream), contextlib.redirect_stdout(stream):
                log_args(logger, base_args)
                log_args(logger, override_args, title='Cli overrides:')

                config = merge(config, override_args, strategy=Strategy.TYPESAFE_REPLACE)
                config_module.update_config(config)
                log_args(logger, config, title='Merged config:')

                set_random_seed(config['general']['seed'],
                                pytorch=args['general']['max_reproducibility'])
                log_params(config)
                main_inner(config, mlflow_run, tmp_dir, logger)
    finally:
        mlflow.log_artifact(str(log_file_path))
        rmtree(tmp_path, 'out.log')

    mlflow.end_run()


def setup_environment(tmp_base: str):
    base_path = Path(tmp_base)
    if not base_path.exists():
        base_path.mkdir()


def main_inner(config: dict, mlflow_run: mlflow.ActiveRun, tmp_dir: str, logger):

    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    log_experiment_info(logger, mlflow_run, config['experiment']['name'])
    track_environment_info(mlflow_run)

    builder = get_active_learner_builder(config, tmp_dir)
    exp = builder.build()
    exp.run(builder.train, builder.valid, builder.test)

    log_mlflow_artifacts(exp.results, tmp_dir)
    log_experiment_info(logger, mlflow_run, config['experiment']['name'])


def log_params(config):
    mlflow.log_param('classifier_name', config['classifier']['classifier_name'])
    classifier_pretrained_model = config['classifier']['classifier_kwargs']['transformer_model'] \
        if config['classifier']['classifier_name'] == 'transformer' else ''
    mlflow.log_param('classifier_pretrained_model', classifier_pretrained_model)
    mlflow.log_param('dataset_name', config['dataset']['dataset_name'])
    mlflow.log_param('query_strategy', config['active_learner']['query_strategy'])

    with TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir).joinpath('config.json')
        with open(config_path, 'w+') as f:
            json.dump(config, f)
        mlflow.log_artifact(str(config_path))


if __name__ == '__main__':
    # import torch
    # torch.autograd.set_detect_anomaly(True)

    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    from importlib import import_module

    args = get_parser().parse_args()
    override_args = get_non_default_args(args)

    config_module = import_module(args['config_module'])
    setup_environment(config_module.TMP_BASE)

    config = Configuration('config')

    experiment = get_experiment(config_module.EXPERIMENT_NAME)
    config.override({
        'experiment': {
            'id': experiment.experiment_id,
            'name': config_module.EXPERIMENT_NAME
        }
    })
    config_module.set_defaults(config, override_args)

    container = Container(config=config)
    config_container = config_module.Container()
    container.override(config_container)

    base = 'active_learning_lab.experiments.active_learning'
    container.wire(modules=[__name__,
                            f'{base}.active_learning_experiment',
                            ])

    main(config_module, args, override_args)
