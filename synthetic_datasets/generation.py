import pandas as pd

from generation_methods.sdv_copula import sdv_copula_generator
from generation_methods.sdv_ctgan import sdv_ctgan_generator
from generation_methods.gmm import GMMSyntheticDataGenerator
from generation_methods.data_synthesizer import data_synthesizer_bayes_generator
from utils.preprocessing import borderline_smote, manual_factorize
from original_datasets.factorize_params import maternal_factorize_params, uci_heart_factorize_params

ahmad_train_df = pd.read_csv('../original_datasets/train_datasets/ahmad_train.csv')
maternal_factorized_train_df = pd.read_csv('../original_datasets/train_datasets/maternal_factorized_train.csv')
uci_train_df = pd.read_csv('../original_datasets/train_datasets/uci_train.csv')
uci_factorized_train_df = pd.read_csv('../original_datasets/train_datasets/uci_factorized_train.csv')

names = [
    "ahmad",
    "maternal_factorized",
    "uci_factorized",
]

targets = [
    'Event',
    'RiskLevel',
    'HeartDisease',
]

datasets = [
    ahmad_train_df,
    maternal_factorized_train_df,
    uci_factorized_train_df,
]

factorize_list = [
    None,
    maternal_factorize_params,
    uci_heart_factorize_params,
]

balances = [
    'border',
    None,
]

num_rows = 1000

if __name__ == '__main__':
    for data_name, target, ds, f_list in zip(names, targets, datasets, factorize_list):
        shape = ds.shape[0]

        for balance in balances:
            if balance == 'border':
                dataset = borderline_smote(ds.copy(), target)
                name = 'border_' + data_name
            else:
                name = data_name
                dataset = ds.copy()

            dataset_1 = dataset.copy()
            gmm_generator = GMMSyntheticDataGenerator()
            gmm_generator.fit(dataset_1, target, factorize_list=f_list)
            gmm_generator.generate(shape, False).to_csv(f'datasets/gmm_{name}.csv', index=False)

            dataset_2 = dataset.copy()
            gmm_1000_generator = GMMSyntheticDataGenerator()
            gmm_1000_generator.fit(dataset_2, target, factorize_list=f_list)
            gmm_1000_generator.generate(num_rows, False).to_csv(f'datasets/gmm_1000_{name}.csv', index=False)

            if name in ["ahmad", "border_ahmad"]:
                dataset_3 = dataset.copy()
                sdv_copula_generator(dataset_3, shape).to_csv(f'datasets/copula_{name}.csv', index=False)
                dataset_4 = dataset.copy()
                sdv_copula_generator(dataset_4, num_rows).to_csv(f'datasets/copula_1000_{name}.csv', index=False)

                dataset_5 = dataset.copy()
                sdv_ctgan_generator(dataset_5, shape).to_csv(f'datasets/ctgan_{name}.csv', index=False)
                dataset_6 = dataset.copy()
                sdv_ctgan_generator(dataset_6, num_rows).to_csv(f'datasets/ctgan_1000_{name}.csv', index=False)

                data_synthesizer_bayes_generator(
                    data_in_csv='../original_datasets/train_datasets/ahmad_train.csv',
                    num_rows=shape,
                    description_file=f'nbs_{name}_bayes_description.json').to_csv(f'datasets/ds_{name}.csv',
                                                                                  index=False)
                data_synthesizer_bayes_generator(
                    data_in_csv='../original_datasets/train_datasets/ahmad_train.csv',
                    num_rows=num_rows,
                    description_file=f'nbs_{name}_1000_bayes_description.json').to_csv(f'datasets/ds_1000_{name}.csv',
                                                                                       index=False)

            elif name in ["maternal_factorized", "border_maternal_factorized"]:
                dataset_7 = dataset.copy()
                sdv_copula_generator(dataset_7, shape, '../original_datasets/maternal_metadata.json').to_csv(
                    f'datasets/copula_{name}.csv', index=False)
                dataset_8 = dataset.copy()
                sdv_copula_generator(dataset_8, num_rows, '../original_datasets/maternal_metadata.json').to_csv(
                    f'datasets/copula_1000_{name}.csv', index=False)

                dataset_9 = dataset.copy()
                sdv_ctgan_generator(dataset_9, shape, '../original_datasets/maternal_metadata.json').to_csv(
                    f'datasets/ctgan_{name}.csv', index=False)
                dataset_10 = dataset.copy()
                sdv_ctgan_generator(dataset_10, num_rows, '../original_datasets/maternal_metadata.json').to_csv(
                    f'datasets/ctgan_1000_{name}.csv', index=False)

                data_synthesizer_bayes_generator(
                    data_in_csv='../original_datasets/train_datasets/maternal_factorized_train.csv',
                    num_rows=shape,
                    description_file=f'nbs_{name}_bayes_description.json').to_csv(f'datasets/ds_{name}.csv',
                                                                                  index=False)
                data_synthesizer_bayes_generator(
                    data_in_csv='../original_datasets/train_datasets/maternal_factorized_train.csv',
                    num_rows=num_rows,
                    description_file=f'nbs_{name}_1000_bayes_description.json').to_csv(f'datasets/ds_1000_{name}.csv',
                                                                                       index=False)
            elif name in ["uci_factorized", "border_uci_factorized"]:
                dataset_11 = dataset.copy()
                sdv_copula_generator(dataset_11, shape, '../original_datasets/uci_metadata.json').to_csv(
                    f'datasets/copula_{name}.csv', index=False)
                dataset_12 = dataset.copy()
                sdv_copula_generator(dataset_12, num_rows, '../original_datasets/uci_metadata.json').to_csv(
                    f'datasets/copula_1000_{name}.csv', index=False)

                dataset_13 = dataset.copy()
                sdv_ctgan_generator(dataset_13, shape, '../original_datasets/uci_metadata.json').to_csv(
                    f'datasets/ctgan_{name}.csv', index=False)
                dataset_14 = dataset.copy()
                sdv_ctgan_generator(dataset_14, num_rows, '../original_datasets/uci_metadata.json').to_csv(
                    f'datasets/ctgan_1000_{name}.csv', index=False)

                data_synthesizer_bayes_generator(
                    data_in_csv='../original_datasets/train_datasets/uci_factorized_train.csv',
                    num_rows=shape,
                    description_file=f'nbs_{name}_bayes_description.json').to_csv(f'datasets/ds_{name}.csv',
                                                                                  index=False)
                data_synthesizer_bayes_generator(
                    data_in_csv='../original_datasets/train_datasets/uci_factorized_train.csv',
                    num_rows=num_rows,
                    description_file=f'nbs_{name}_1000_bayes_description.json').to_csv(f'datasets/ds_1000_{name}.csv',

                                                                                       index=False)
    name = "un_processed_uci"
    target = 'HeartDisease'
    shape = uci_train_df.shape[0]

    dataset = uci_train_df.copy()

    gmm_1000_generator = GMMSyntheticDataGenerator()
    gmm_1000_generator.fit(dataset, target)
    manual_factorize(gmm_1000_generator.generate(num_rows), uci_heart_factorize_params, clip=False,
                     input_fact=True).to_csv(f'datasets/gmm_1000_{name}.csv', index=False)

    gmm_generator = GMMSyntheticDataGenerator()
    gmm_generator.fit(dataset, target)
    manual_factorize(gmm_generator.generate(shape), uci_heart_factorize_params, clip=False, input_fact=True).to_csv(
        f'datasets/gmm_{name}.csv', index=False)

    dataset = uci_train_df.copy()
    data_123 = manual_factorize(sdv_copula_generator(dataset, shape),
                                factorize_list=uci_heart_factorize_params, clip=False, input_fact=True)
    data_123.to_csv(f'datasets/copula_{name}.csv', index=False)

    dataset = uci_train_df.copy()
    data_1234 = manual_factorize(sdv_copula_generator(dataset, num_rows),
                                 factorize_list=uci_heart_factorize_params, clip=False, input_fact=True)
    data_1234.to_csv(f'datasets/copula_1000_{name}.csv', index=False)

    dataset = uci_train_df.copy()
    manual_factorize(sdv_ctgan_generator(dataset, shape),
                     factorize_list=uci_heart_factorize_params, clip=False, input_fact=True).to_csv(
        f'datasets/ctgan_{name}.csv', index=False)
    dataset = uci_train_df.copy()
    manual_factorize(sdv_ctgan_generator(dataset, num_rows),
                     factorize_list=uci_heart_factorize_params, clip=False, input_fact=True).to_csv(
        f'datasets/ctgan_1000_{name}.csv', index=False)
