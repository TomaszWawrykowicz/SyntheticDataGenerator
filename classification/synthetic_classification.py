import os
import numpy as np
import time

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

names = [
    "ahmad",
    'maternal',
    'uci',
]

clf = RandomForestClassifier()

ahmad_rand_forest_params_grid = {'clf__n_estimators': [200],
                                 'clf__criterion': ['gini'],
                                 'clf__min_samples_split': [2],
                                 'clf__max_features': ['sqrt'],
                                 'clf__class_weight': [None]}

maternal_rand_forest_params_grid = {'clf__n_estimators': [50],
                                    'clf__criterion': ['entropy'],
                                    'clf__min_samples_split': [2],
                                    'clf__max_features': ['log2'],
                                    'clf__class_weight': [None]}

uci_rand_forest_params_grid = {'clf__n_estimators': [100],
                               'clf__criterion': ['log_loss'],
                               'clf__min_samples_split': [3],
                               'clf__max_features': ['log2'],
                               'clf__class_weight': ['balanced']}

params = [
    ahmad_rand_forest_params_grid,
    maternal_rand_forest_params_grid,
    uci_rand_forest_params_grid,
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

datasets_folders = [
    '../synthetic_datasets/datasets/ahmad/',
    '../synthetic_datasets/datasets/maternal/',
    '../synthetic_datasets/datasets/uci/',
]


def grid_function(data, target, name, classifier, param_grid, x_test, y_test, txt_name):
    x = data.drop(target, axis=1).values
    y = np.ravel(data[target])

    classifier = Pipeline([('scale', StandardScaler()),
                           ('clf', classifier)])
    grid = GridSearchCV(classifier, param_grid=param_grid, scoring='f1_macro', cv=5)
    grid.fit(x, y)
    train_score = grid.score(x, y)
    print(grid.best_estimator_, train_score)
    score = grid.score(x_test, y_test)
    print('Score', score)

    with open(f'{txt_name}_synthetic_grid_results.txt', 'a') as f:
        f.write(f'Classifier Random Forest')
        f.write(f'\nOn dataset: {name}')
        f.write(f'\nParameters: {grid.best_params_}')
        f.write(f'\nBest score: {train_score}\n')

    return grid.best_estimator_, train_score, score


for name, target, param_grid, test_data, datasets_folder in zip(names, targets, params, test_datas, datasets_folders):
    test_data = pd.read_csv(test_data)
    x_test = test_data.drop(target, axis=1)
    y_test = np.ravel(test_data[target])

    datasets = os.listdir(datasets_folder)
    for dataset_path in datasets:
        start_time = time.time()
        dataset_name = os.path.splitext(dataset_path)[0]
        dataset = pd.read_csv(datasets_folder + dataset_path)

        estimator, estimator_best_score, best_score = grid_function(data=dataset, target=target, name=dataset_name,
                                                                    classifier=clf, param_grid=param_grid,
                                                                    x_test=x_test, y_test=y_test, txt_name=name)

        # score = estimator.score(x_test, y_test)
        end_time = time.time()

        with open(f'{name}_classification.txt', 'a') as res:
            res.write(f'Classifier: Random Forest')
            res.write(f'\nTrained on: {dataset_name}')
            res.write(f'\nScore on train data: {estimator_best_score}')
            res.write(f'\nScore on test data: {best_score}\n')
            res.write(f'Execution time: {end_time - start_time}\n\n')
