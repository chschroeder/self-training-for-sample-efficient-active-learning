import mlflow

from pathlib import Path
from mlflow.entities import Experiment


def get_experiment(experiment_name: str) -> Experiment:
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError('No mlflow experiments with name \'{}\' exists. '
                         'Please create the experiment first.'.format(experiment_name))

    return experiment


def get_mlflow_tmp_path(run: mlflow.ActiveRun) -> Path:
    """
    Return tmp dir relative to a mlflow active run.

    Parameters
    ----------
    run : mlflow.ActiveRun
        an mlflow.ActiveRun object

    Returns
    -------
    path : Path
        path to a tmp directory relative to the current run directory
    """

    base_path = run.info.artifact_uri
    if ':' in base_path:
        base_path = base_path[run.info.artifact_uri.index(':') + 1:]
    base_path = Path(base_path).joinpath('..')
    tmp_path = base_path.joinpath('tmp/').resolve()

    if not tmp_path.exists():
        tmp_path.resolve().mkdir()

    return tmp_path
