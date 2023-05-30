import pandas as pd

from sklearn.model_selection import train_test_split

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
