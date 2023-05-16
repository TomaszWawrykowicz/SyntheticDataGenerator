from decimal import Decimal

import numpy as np
import pandas as pd
import tqdm
from numpy import int64, int32, float64
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture

# Option to display all columns
pd.set_option('display.max_columns', None)


class GMMSyntheticDataGenerator(BaseEstimator):

    def __init__(self):
        self.n_features = None
        self.n_samples = None

        self.original_data = None
        self.data = None

        self.model = None
        self.factorize = []

        self.original_dtypes = None
        self.dtypes = None
        self.names = None

        self.means = None
        self.covariances = None
        self.precision_cholesky = None
        self.weights = None

    def preproccess_data(self, data):
        self.original_dtypes = data.dtypes.to_dict()
        self.names = self.original_dtypes.keys()

        for (columnName, columnData) in data.items():

            if columnData.dtype == object:
                factorized_data, indexes = pd.factorize(data[columnName])
                data[columnName] = factorized_data
                self.factorize.append((columnName, {i: indexes[i] for i in range(len(indexes))}))

        self.dtypes = data.dtypes.to_dict()
        print('Data preproccessed')
        print(self.factorize)

    def postprocess_data(self, rounding=True, to_integers=True, mapping=False):
        # Rounding and mapping original data naming

        # Rounding
        # if int -> int, dec = 0, factorize niżej, float
        if rounding:
            for key, dtype in self.dtypes.items():
                if dtype in [int64, int32]:
                    self.data[key] = self.data[key].round(0).astype('int64')
                elif dtype == float64:
                    dec = max([abs(Decimal(str(x)).as_tuple().exponent)
                               for x in self.original_data[key].sample(10).to_list()])
                    self.data[key] = self.data[key].round(dec).astype('float64')
                elif dtype == object and to_integers:
                    self.data[key] = self.data[key].round(0).astype('int64')

        # Restoring discrete values
        for column in self.factorize:
            # print(self.data[column[0]])
            # print('klucze', list(column[1].keys()))

            lenght = len(column[1])
            self.data[column[0]] = self.data[column[0]].apply(
                lambda x: 0 if x < 0 else (lenght - 1 if x >= lenght else x))
            # np.clip

            # Mapping back
            if mapping:
                self.data[column[0]] = self.data[column[0]].map(column[1])
                # print(self.data[column[0]])
                # print('Mapped original data types')

    def fit(self, x):
        # self.data = x
        self.original_data = x

        self.preproccess_data(self.original_data)
        # self.postprocess_data()
        print('Start building model')
        # print(type(self.data))

        models = []
        bic_scores = []

        n_components_range = range(1, 41)

        for n in tqdm.tqdm(n_components_range):
            model = GaussianMixture(n, covariance_type='full').fit(self.original_data)
            models.append(model)
            bic_scores.append(model.bic(self.original_data))

        best_n_components = n_components_range[np.argmin(bic_scores)]
        print('Number of components:', best_n_components, 'based on Bayes Information criterion')
        # print('test', models[best_n_components].bic(self.data), bic_scores[best_n_components])

        self.model = models[n_components_range[np.argmin(bic_scores)]]
        print('Model built')
        # print(self.model.sample(10)[0])

    def predict(self):
        pass

    def sample(self, n_sample):
        if self.model:
            self.data = pd.DataFrame(self.model.sample(n_sample)[0], columns=self.names)
            # self.postprocess_data(rounding=True, to_integers=False, mapping=False)
            self.postprocess_data(rounding=True, to_integers=True, mapping=True)
            print(self.original_data.info())
            print(self.data.info())
            print(self.original_data.describe())
            print(self.data.describe())
            return self.data
        else:
            raise TypeError('Model not build')

    def save_model(self):
        np.save('gmm_names', list(self.names), allow_pickle=True)
        np.save('gmm_weights', self.model.weights_, allow_pickle=False)
        np.save('gmm_precisions_cholesky', self.model.precisions_cholesky_, allow_pickle=False)
        np.save('gmm_means', self.model.means_, allow_pickle=False)
        np.save('gmm_covariances', self.model.covariances_, allow_pickle=False)
        print('Model saved successfully')

    def load(self, names, means, covariances, precision_cholesky, weights):
        if isinstance(names, list):
            self.names = names
        elif isinstance(names, str) and names.endswith('.npy'):
            self.names = np.load(names)
        else:
            raise ValueError('Names argument is not a list or .npy file')
        print(self.names)
        means = np.load(means)
        covar = np.load(covariances)

        self.model = GaussianMixture(n_components=len(means), covariance_type='full')
        self.model.precisions_cholesky = np.load(precision_cholesky)
        self.model.weights_ = np.load(weights)
        self.model.means_ = means
        self.model.covariances_ = covar
        print('Model loaded successfully')

    def save_data(self, name):
        self.data.to_csv(name, index=False)


# generator = SyntheticDataGenerator(2, 5)
# print(generator.get_params())  # {'x': 2, 'y': 5}
# generator.set_params(**{'y': 3, 'x': 6})
# print(generator.get_params())  # {'x': 6, 'y': 3}


gen = GMMSyntheticDataGenerator()
kaggle_data = pd.read_csv('heart_uci.csv')
gen.fit(kaggle_data)
syn_data = gen.sample(918)
print(syn_data)
# gen.save_model()
gen.save_data('new_data.csv')

# gen.load('gmm_names.npy', 'gmm_means.npy', 'gmm_covariances.npy', 'gmm_precisions_cholesky.npy', 'gmm_weights.npy')
# arr = gen.sample(100)
# print(arr)  # DO POPRAWY, BO MUSI ŁADOWAĆ TWORZONY GENERATOR A NIE BAZOWY
