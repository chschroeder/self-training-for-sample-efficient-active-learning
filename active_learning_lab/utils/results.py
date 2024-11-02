"""Methods for aggregating or evaluating results, or helpers therefore."""
import json
import logging
import re
import warnings

import mlflow

import numpy as np
import pandas as pd

from pathlib import Path
from mlflow import get_experiment_by_name, search_runs


logger = logging.getLogger(__name__)


def get_results(exp_name: str):
    experiment = get_experiment_by_name(exp_name)
    experiment_id = experiment.experiment_id

    runs = search_runs(experiment_ids=[experiment_id])
    runs['experiment_name'] = [mlflow.get_run(run_id).info.run_name for run_id in runs.run_id.values]
    cols_reordered = np.concatenate((runs.columns[:2].values, [runs.columns[-1]], runs.columns[2:].values[:-1]))
    runs = runs[cols_reordered]
    return runs[(~runs.run_id.isna()) & (runs.status == 'FINISHED')]


def check_for_duplicates(df, columns=['params.dataset_name', 'params.classifier_name', 'params.query_strategy']):
    df_tmp = df.groupby(columns).size().reset_index()

    duplicates = df_tmp[df_tmp[0] > 1]

    if len(duplicates) > 0:
        duplicates = df.set_index(columns).loc[duplicates.itertuples(index=False, name=None)]

        duplicates = duplicates.groupby(columns).agg({'experiment_name': list}).reset_index()
        duplicates = duplicates.rename(columns={'experiment_name': 'experiment_names'})

        # TODO: columns must refer to string columns
        error_cases = ['\t' + '_'.join([row[col] for col in columns]) + ': ' + str(row['experiment_names'])
                       for _, row in duplicates.iterrows()]
        error_cases = '\n'.join(error_cases)

        raise ValueError('Multiple results were found for the following strategies:\n'
                         f'{error_cases}')

    return df


def check_for_duplicates_take_first(df, columns=['params.dataset_name', 'params.classifier_name', 'params.query_strategy']):
    df = df.sort_values('start_time', ascending=False)
    df_tmp = df.groupby(columns).size().reset_index()

    duplicates = df_tmp[df_tmp[0] > 1]

    if len(duplicates) > 0:
        duplicates = df.set_index(columns).loc[duplicates.itertuples(index=False, name=None)]

        duplicates = duplicates.groupby(columns).agg({'run_id': list}).reset_index()
        duplicates = duplicates.rename(columns={'run_id': 'run_ids'})

        # TODO: columns must refer to string columns
        error_cases = ['\t' + '_'.join([str(row[col]) for col in columns]) + ': ' + str(row['run_ids'])
                       for _, row in duplicates.iterrows()]

        for error in error_cases:
            warnings.warn(f'Multiple entries found for configuration: {error}')

    return df.groupby(columns).first()


# TODO: this can be improved by extending assemble_dfs to include all parameters
def check_for_reproduciblity(df):

    for row in df.itertuples():
        registry_uri = re.sub('^file://', '', mlflow.get_registry_uri())
        registry_uri += f'/{row.experiment_id}/{row.run_id}'

        config_path = Path(registry_uri).joinpath('artifacts/config.json')

        try:
            config = json.load(config_path.open())
            if not config['general']['max_reproducibility']:
                logger.warning(f'{row.run_id} ({row.params.classifier_name} / {row.params.dataset_name} / '
                               f'{row.params.query_strategy}) has not set the max_reproducibility flag')
        except FileNotFoundError:
            # raised when experiment still running or aborted
            logger.warning(f'Run {row.run_id} has no params/config (still running or aborted)')

    return df


def assemble_df(df, csv_name, index_col=None):
    dfs = []

    for row in df.itertuples():
        registry_uri = re.sub('^file://', '', mlflow.get_registry_uri())
        registry_uri += f'/{row.experiment_id}/{row.run_id}'

        try:
            csv_path = Path(registry_uri).joinpath(f'artifacts/{csv_name}')

            df_tmp = pd.read_csv(csv_path, header=0, index_col=index_col)

            # +1 because row._fields has an additional field "Index" at the beginning
            df_tmp['dataset_name'] = row[df.columns.get_loc('params.dataset_name')+1]
            df_tmp['classifier'] = row[df.columns.get_loc('params.classifier_name')+1]
            df_tmp['query_strategy'] = row[df.columns.get_loc('params.query_strategy')+1]

            dfs.append(df_tmp)
        except FileNotFoundError:
            # raised when experiment still running or aborted
            logger.warning(f'folder {row.run_id} has no params/config (still running or aborted)')

    df_all = pd.concat(dfs)
    return df_all
