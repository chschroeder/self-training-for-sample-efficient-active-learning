import logging
import mlflow

import numpy as np
import pandas as pd
from sklearn.metrics import auc


def build_auc_metrics(tmp_dir, metrics_tracker, artifacts):

    df_auc = compute_auc_from_metrics_df(metrics_tracker.measured_metrics)
    auc_csv = tmp_dir.joinpath('auc.csv').resolve()
    df_auc.to_csv(auc_csv, index=False, header=True)
    artifacts += [('auc.csv', auc_csv)]

    df_auc_acc = df_auc.groupby(lambda _: True).agg(lambda x: [np.mean(x), np.std(x, ddof=0)])
    auc_agg_csv = tmp_dir.joinpath('auc_agg.csv').resolve()
    df_auc_acc.columns = df_auc_acc.columns.to_flat_index()
    df_auc_acc.columns = [tup[0] + '_' + tup[1] for tup in df_auc_acc.columns]
    df_auc_acc.to_csv(auc_agg_csv, index=False, header=True)
    artifacts += [('auc_agg.csv', auc_csv)]

    log_and_track_aggregate_metrics(df_auc, metrics_tracker)

    return artifacts


def log_and_track_aggregate_metrics(df_auc, metrics_tracker):

    auc_test_acc_mean = df_auc['auc_test_acc'].mean()
    auc_test_acc_std = df_auc['auc_test_acc'].std(ddof=0)
    auc_test_micro_f1_mean = df_auc['auc_test_micro_f1'].mean()
    auc_test_micro_f1_std = df_auc['auc_test_micro_f1'].std(ddof=0)

    logging.info('#--------------------------------')
    logging.info(f'AUC: {auc_test_acc_mean:.4f} (+/- {auc_test_acc_std:.4f})')

    mlflow.log_metric('auc_test_acc_mean', auc_test_acc_mean)
    mlflow.log_metric('auc_test_acc_std', auc_test_acc_std)
    mlflow.log_metric('auc_test_micro_f1_mean', auc_test_micro_f1_mean)
    mlflow.log_metric('auc_test_micro_f1_std', auc_test_micro_f1_std)

    query_id = metrics_tracker.measured_metrics['query_id'].max()
    query_rows = metrics_tracker.measured_metrics[
        metrics_tracker.measured_metrics['query_id'] == query_id]

    test_acc_mean = query_rows['test_acc'].mean()
    test_acc_std = query_rows['test_acc'].std(ddof=0)
    test_micro_f1_mean = query_rows['test_micro_f1'].mean()
    test_micro_f1_std = query_rows['test_micro_f1'].std(ddof=0)
    test_macro_f1_mean = query_rows['test_macro_f1'].mean()
    test_macro_f1_std = query_rows['test_macro_f1'].std(ddof=0)

    mlflow.log_metric('test_acc_mean', test_acc_mean)
    mlflow.log_metric('test_acc_std', test_acc_std)
    mlflow.log_metric('test_micro_f1_mean', test_micro_f1_mean)
    mlflow.log_metric('test_micro_f1_std', test_micro_f1_std)
    mlflow.log_metric('test_macro_f1_mean', test_macro_f1_mean)
    mlflow.log_metric('test_macro_f1_std', test_macro_f1_std)

    logging.info(f'ACC: {test_acc_mean:.4f} (+/- {test_acc_std:.4f})')
    logging.info(f'mF1: {test_micro_f1_mean:.4f} (+/- {test_micro_f1_std:.4f})')
    logging.info('#--------------------------------')


def compute_auc_from_metrics_df(df_metrics):

    def f_agg(df):
        auc_x = df['num_samples'].tolist()
        span_x = (auc_x[-1] - auc_x[0])

        auc_y_acc = df['test_acc'].tolist()
        auc_y_f1 = df['test_micro_f1'].tolist()

        df_result = pd.DataFrame([
            [auc(auc_x, auc_y_acc) / span_x, auc(auc_x, auc_y_f1) / span_x]
        ], columns=['auc_test_acc', 'auc_test_micro_f1'])

        return df_result

    df_result = df_metrics.groupby('run_id').apply(f_agg)
    df_result.index = df_result.index.levels[0]

    return df_result
