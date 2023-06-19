import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import KFold, train_test_split


def one_column_factorize(column):
    return column.factorize()[0]


def columns_factorize(df):
    for (columnName, columnData) in df.items():
        if columnData.dtype == object:
            df[columnName] = df[columnName].factorize()[0]
    return df


def semi_auto_factorize(data):
    factorize_list = []
    for (columnName, columnData) in data.items():
        if columnData.dtype == object:
            factorized_data, indexes = pd.factorize(data[columnName])
            data[columnName] = factorized_data
            factorize_list.append((columnName, {i: indexes[i] for i in range(len(indexes))}))
    return data, factorize_list


def manual_factorize(data, factorize_list, clip=True, input_fact=False):
    """
    Func for factorize dataset
    :param clip: default True, apply np.clip on data, set False for iterate over string values
    :param data: dataset
    :param factorize_list: list of tuples containing column name and mapping dict, [('Sex', {0: 'M', 1: 'F'}),
    ('BodyType', {0: 'Slim', 1: 'Fat'})]
    :param input_fact: default False, set True for factorize input data
    :return data: factorized dataset
    """
    for column in factorize_list:
        if clip:
            length = len(column[1])
            # data[column[0]] = data[column[0]].apply(
            #     lambda x: 0 if x < 0 else (length - 1 if x >= length else x))
            data[column[0]] = np.clip(data[column[0]], 0, length - 1)
        if input_fact:
            data[column[0]] = data[column[0]].map(column[1])
        # else:
        #     data[column[0]] = data[column[0]].map(column[1])
    return data


def cross_validation(data, file_name, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)

    for i, split_index in enumerate(kfold.split(data)):
        data.iloc[split_index[0]].to_csv(f'train_{i + 1}_{file_name}.csv', index=False)
        data.iloc[split_index[1]].to_csv(f'test_{i + 1}_{file_name}.csv', index=False)
    return True


def simple_split(data, test_size, shuffle=True, random_state=42):
    """
    Func for splitting dataset into train and test datasets
    :param data:
    :param test_size:
    :param shuffle: set False to prevent dataset from shuffled
    :param random_state: set None to generate split datasets randomly
    :return: split into train and test datasets
    """
    x, y = train_test_split(data, test_size=test_size, shuffle=shuffle, random_state=random_state)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return x, y


def borderline_smote(data, target):
    border_smote = BorderlineSMOTE()
    x = data.drop(target, axis=1)
    y = data[target]

    x, y = border_smote.fit_resample(x, y)

    x[target] = y

    return x
