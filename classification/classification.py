import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from utils.preprocessing import manual_factorize
from original_datasets.factorize_params import uci_heart_factorize_params, maternal_factorize_params

warnings.filterwarnings('ignore')

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]


def grid_function(data, target, name, classifier, param_grid, x_test, y_test):
    x = data.drop(target, axis=1).values
    y = np.ravel(data[target])

    classifier = Pipeline([('scale', StandardScaler()),
                           ('clf', classifier)])
    grid = GridSearchCV(classifier, param_grid=param_grid, scoring='f1_weighted', cv=5)
    grid.fit(x, y)
    train_score = grid.score(x, y)
    print(grid.best_estimator_, train_score)
    score = grid.score(x_test, y_test)
    print('score', score)

    with open('grid_results.txt', 'a') as f:
        f.write(f'Classifier {name}')
        f.write(f'\nParameters: {grid.best_params_}')
        f.write(f'\nBest score: {train_score}\n')

    return grid.best_estimator_, train_score, score


kn_params_grid = {'clf__algorithm': ['ball_tree', 'kd_tree', 'brute'],
                  'clf__n_neighbors': [3, 5, 10],
                  'clf__weights': ['uniform', 'distance'],
                  'clf__leaf_size': [20, 30, 40],
                  'clf__p': [1, 2]}
linear_svm_params_grid = {'clf__C': [0.01, 1, 1000],
                          'clf__gamma': ['scale', 'auto'],
                          'clf__tol': [0.0001, 0.001, 0.01],
                          'clf__class_weight': [None, 'balanced']}
rbf_svm_params_grid = {'clf__C': [0.01, 1, 1000],
                       'clf__gamma': ['scale', 'auto'],
                       'clf__tol': [0.0001, 0.001, 0.01],
                       'clf__class_weight': [None, 'balanced']}
gauss_process_params_grid = {'clf__n_restarts_optimizer': [0, 1, 2],
                             'clf__max_iter_predict': [50, 100, 200]}
dec_tree_params_grid = {'clf__criterion': ['gini', 'entropy', 'log_loss'],
                        'clf__splitter': ['best', 'random'],
                        'clf__min_samples_split': [2, 3],
                        'clf__max_features': ['sqrt', 'log2', None],
                        'clf__class_weight': [None, 'balanced']}
rand_forest_params_grid = {'clf__n_estimators': [50, 100, 200],
                           'clf__criterion': ['gini', 'entropy', 'log_loss'],
                           'clf__min_samples_split': [2, 3],
                           'clf__max_features': ['sqrt', 'log2', None],
                           'clf__class_weight': [None, 'balanced', 'balanced_subsample']}
mlpc_random_grid = {'clf__hidden_layer_sizes': [(50,), (100,), (125,)],
                    'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'clf__solver': ['lbfgs', 'sgd', 'adam'],
                    'clf__alpha': [0.0001, 0.001],
                    'clf__learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'clf__learning_rate_init': [0.001, 0.01, 0.1],
                    # 'clf__max_iter': [200, 400, 500],
                    'clf__max_iter': [500],
                    'clf__max_fun': [10000, 15000, 20000]}
ada_boost_params_grid = {'clf__n_estimators': [25, 50, 100],
                         'clf__learning_rate': [0.1, 1, 10]}
gauss_nb_params_grid = {'clf__var_smoothing': [1e-10, 1e-9, 1e-8]}
qda_params_grid = {}

params = [
    kn_params_grid,
    linear_svm_params_grid,
    rbf_svm_params_grid,
    gauss_process_params_grid,
    dec_tree_params_grid,
    rand_forest_params_grid,
    mlpc_random_grid,
    ada_boost_params_grid,
    gauss_nb_params_grid,
    qda_params_grid,
]

# target = 'Event'
# train_data = pd.read_csv('../original_datasets/train_datasets/ahmad_train.csv')
# test_data = pd.read_csv('../original_datasets/test_datasets/ahmad_test.csv')

target = 'HeartDisease'
train_data = pd.read_csv('../original_datasets/train_datasets/uci_factorized_train.csv')
test_data = pd.read_csv('../original_datasets/test_datasets/uci_factorized_test.csv')

# target = 'RiskLevel'
# train_data = pd.read_csv('../original_datasets/train_datasets/maternal_factorized_train.csv')
# test_data = pd.read_csv('../original_datasets/test_datasets/maternal_factorized_test.csv')

x_test = test_data.drop(target, axis=1)
y_test = np.ravel(test_data[target])

for name, clf, param_grid in zip(names, classifiers, params):
    start_time = time.time()
    estimator, estimator_best_score, best_score = grid_function(data=train_data, target=target, name=name,
                                                                classifier=clf,
                                                                param_grid=param_grid, x_test=x_test, y_test=y_test)

    # score = estimator.score(x_test, y_test)
    end_time = time.time()

    with open('classification.txt', 'a') as res:
        res.write(f'Classifier: {name}')
        res.write(f'\nScore on train data: {estimator_best_score}')
        res.write(f'\nScore on test data: {best_score}\n')
        res.write(f'Execution time: {end_time - start_time}\n\n')
