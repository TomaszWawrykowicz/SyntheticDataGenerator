import numpy as np
import pandas as pd
import tqdm
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture


class GMMSyntheticDataGenerator(BaseEstimator):

    def __init__(self):
        self.n_features = None
        self.n_samples = None
        self.data = None
        self.model = None
        self.factorize = []

    def preproccess_data(self):
        for (columnName, columnData) in self.data.items():

            if columnData.dtype == object:
                factorized_data, indexes = pd.factorize(self.data[columnName])
                self.data[columnName] = factorized_data
                self.factorize.append((columnName, {i: indexes[i] for i in range(len(indexes))}))
        print('Data preproccessed')
        print(self.factorize)

    def postprocess_data(self, rounding=False, to_integers=False, mapping=False):
        # Rounding and mapping original data naming
        # Rounding
        # if int -> int, dec = 0, factorize niżej, float nie wiem
        if rounding:
            pass

        for column in self.factorize:
            print(self.data[column[0]])
            print('klucze', list(column[1].keys()))
            #to_integers
            if to_integers:
                pass

            # Mapping back
            if mapping:
                self.data[column[0]] = self.data[column[0]].map(column[1])
            print(self.data[column[0]])
        print('Mapped original data types')

    def fit(self, x):
        self.data = x

        self.preproccess_data()
        self.postprocess_data()
        return False
        print('Start building model')
        # print(type(self.data))

        models = []
        bic_scores = []

        n_components_range = range(1, 41)

        for n in tqdm.tqdm(n_components_range):
            model = GaussianMixture(n, covariance_type='full').fit(self.data)
            models.append(model)
            bic_scores.append(model.bic(self.data))

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
            return self.model.sample(n_sample)[0]
        else:
            raise TypeError('Model not build')

    def save_model(self):
        np.save('model_weights', self.model.weights_, allow_pickle=False)
        np.save('model_precisions_cholesky', self.model.precisions_cholesky_, allow_pickle=False)
        np.save('model_means', self.model.means_, allow_pickle=False)
        np.save('model_covariances', self.model.covariances_, allow_pickle=False)


# generator = SyntheticDataGenerator(2, 5)
# print(generator.get_params())  # {'x': 2, 'y': 5}
# generator.set_params(**{'y': 3, 'x': 6})
# print(generator.get_params())  # {'x': 6, 'y': 3}


gen = GMMSyntheticDataGenerator()
data = pd.read_csv('heart_uci.csv')
gen.fit(data)
# syn_data = gen.sample(100)
# print(syn_data)
