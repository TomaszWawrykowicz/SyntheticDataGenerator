import pandas as pd

from sklearn.model_selection import train_test_split

from utils.preprocessing import manual_factorize
from factorize_params import maternal_factorize_params, uci_heart_factorize_params

uci_heart_df = pd.read_csv('../original_datasets/uci_heart_disease.csv')
maternal_df = pd.read_csv('../original_datasets/maternal_heart_risk.csv')
ahmad_hear_df = pd.read_csv('../original_datasets/ahmad_heart_failure.csv')


def make_datasets(data, name):
    train, test = train_test_split(data, test_size=0.3)

    train.to_csv(f'train_datasets/{name}_train.csv', index=False)
    test.to_csv(f'test_datasets/{name}_test.csv', index=False)


make_datasets(ahmad_hear_df, 'ahmad')
make_datasets(uci_heart_df, 'uci')
make_datasets(maternal_df, 'maternal')

uci_train = pd.read_csv('../original_datasets/train_datasets/uci_train.csv')
uci_test = pd.read_csv('../original_datasets/test_datasets/uci_test.csv')
maternal_train = pd.read_csv('../original_datasets/train_datasets/maternal_train.csv')
maternal_test = pd.read_csv('../original_datasets/test_datasets/maternal_test.csv')

manual_factorize(uci_train, uci_heart_factorize_params, clip=False, back=True).to_csv(
    'train_datasets/uci_factorized_train.csv', index=False)
manual_factorize(uci_test, uci_heart_factorize_params, clip=False, back=True).to_csv(
    'test_datasets/uci_factorized_test.csv', index=False)

manual_factorize(maternal_train, maternal_factorize_params, clip=False, back=True).to_csv(
    'train_datasets/maternal_factorized_train.csv', index=False)
manual_factorize(maternal_test, maternal_factorize_params, clip=False, back=True).to_csv(
    'test_datasets/maternal_factorized_test.csv', index=False)
