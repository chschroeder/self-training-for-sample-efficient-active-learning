import os
import re

import mlflow

from pathlib import Path
try:
    from pip._internal.operations import freeze
except ImportError:
    from pip.operations import freeze

from active_learning_lab.experiments.active_learning.results import ActiveLearningExperimentArtifacts
from active_learning_lab.utils.mlflow import get_mlflow_tmp_path


def log_mlflow_artifacts(exp_results: ActiveLearningExperimentArtifacts, tmp_path) -> None:
    for name, file in exp_results.artifacts:
        if '/' in name:
            artifact_dir = re.sub('^file:', '', mlflow.get_artifact_uri())
            basedir = Path(artifact_dir).joinpath(name).parents[0]
            if not basedir.exists():
                basedir.mkdir()
            file_rel = str(file.relative_to(tmp_path))
            mlflow.log_artifact(file, file_rel[:file_rel.rindex('/')])
        else:
            mlflow.log_artifact(file)


def track_environment_info(mlflow_run):
    tmp_path = get_mlflow_tmp_path(mlflow_run)
    output_path = tmp_path.joinpath('requirements.txt').resolve()

    with open(output_path, 'w') as f:
        for dep in freeze.freeze():
            f.write(f'{dep}\n')
    mlflow.log_artifact(str(output_path))

    output_path = tmp_path.joinpath('env.txt').resolve()
    with open(output_path, 'w') as f:
        for key, val in os.environ.items():
            f.write(f'{key}\t{val}\n')
    mlflow.log_artifact(str(output_path))
