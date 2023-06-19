import os
import pandas as pd

from evaluation.mmd import mmd_rbf, mmd_linear
from evaluation.hellinger_distance import hellinger_distance
from evaluation.kl_divergence import kl_div
from evaluation.ks_test import ks_test
from evaluation.pcd import PCD

names = [
    "ahmad",
    'maternal',
    'uci',
]

datasets_folders = [
    '../synthetic_datasets/datasets/ahmad/',
    '../synthetic_datasets/datasets/maternal/',
    '../synthetic_datasets/datasets/uci/',
]

targets = [
    'Event',
    'RiskLevel',
    'HeartDisease',
]

test_datas = [
    '../original_datasets/test_datasets/ahmad_test.csv',
    '../original_datasets/test_datasets/maternal_factorized_test.csv',
    '../original_datasets/test_datasets/uci_factorized_test.csv',
]

metrics_names = [
    'pcd',
    'kld',
    'hd',
    'ks_test',
    'mmd_rbf',
    'mmd_linear'
]

metrics = [
    PCD,
    kl_div,
    hellinger_distance,
    ks_test,
    mmd_rbf,
    mmd_linear,
]

train_datas = [
    '../original_datasets/train_datasets/ahmad_train.csv',
    '../original_datasets/train_datasets/maternal_factorized_train.csv',
    '../original_datasets/train_datasets/uci_factorized_train.csv',
]

for name, target, test_data, datasets_folder, train_data in zip(names, targets, test_datas, datasets_folders,
                                                                train_datas):
    test_data = pd.read_csv(test_data)
    train_data = pd.read_csv(train_data)

    numeric_df = pd.DataFrame(columns=['metric', 'dataset', 'value'])
    tabular_metrics = pd.DataFrame(columns=['metric', 'dataset'] + list(test_data.columns))

    for metric_name, metric in zip(metrics_names, metrics):
        score = metric(test_data, train_data)
        try:
            if float(score):
                numeric_df = pd.concat([numeric_df, pd.DataFrame({'metric': [metric_name],
                                                                  'dataset': [name], 'value': [score]})])
                with open(f'of_{name}_metrics.txt', 'a') as res:
                    res.write(f'Metric: {metric_name}')
                    res.write(f'\nSynthetic data version: {name}')
                    res.write(f'\nScore on test data: {score}\n\n\n')
        except TypeError:
            series = pd.DataFrame([[metric_name, name] + [value for value in score.values()]],
                                  columns=list(tabular_metrics.columns))
            tabular_metrics = pd.concat([tabular_metrics, series])

            with open(f'of_{name}_column_metrics.txt', 'a') as res:
                res.write(f'Metric: {metric_name}')
                res.write(f'\nSynthetic data version: {name}')
                res.write(f'\nScore on test data: {score}\n\n\n')

    datasets = os.listdir(datasets_folder)
    for dataset_path in datasets:

        dataset_name = os.path.splitext(dataset_path)[0]
        dataset = pd.read_csv(datasets_folder + dataset_path)
        dataset = dataset.reindex(columns=list(test_data.columns))
        for metric_name, metric in zip(metrics_names, metrics):
            score = metric(test_data, dataset)
            try:
                if float(score):
                    numeric_df = pd.concat([numeric_df, pd.DataFrame({'metric': [metric_name],
                                                                      'dataset': [dataset_name], 'value': [score]})])
                    with open(f'of_{name}_metrics.txt', 'a') as res:
                        res.write(f'Metric: {metric_name}')
                        res.write(f'\nSynthetic data version: {dataset_name}')
                        res.write(f'\nScore on test data: {score}\n\n\n')
            except TypeError:
                series = pd.DataFrame([[metric_name, dataset_name] + [value for value in score.values()]],
                                      columns=list(tabular_metrics.columns))
                tabular_metrics = pd.concat([tabular_metrics, series])

                with open(f'of_{name}_column_metrics.txt', 'a') as res:
                    res.write(f'Metric: {metric_name}')
                    res.write(f'\nSynthetic data version: {dataset_name}')
                    res.write(f'\nScore on test data: {score}\n\n\n')

    numeric_df.sort_values(by=['metric', 'value'], ascending=True).to_csv(f'of_{name}_mmd_pcd.csv', index=False)
    numeric_df.sort_values(by=['metric', 'value'], ascending=True).to_excel(f'of_{name}_mmd_pcd.xlsx', index=False)
    tabular_metrics.sort_values(by=['metric', 'dataset'], ascending=True).to_csv(f'of_{name}_hs_kld_ks.csv', index=False)
    tabular_metrics.sort_values(by=['metric', 'dataset'], ascending=True).to_excel(f'of_{name}_hs_kld_ks.xlsx',
                                                                                   index=False)
