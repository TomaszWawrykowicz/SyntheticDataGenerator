import numpy as np
import pandas as pd
import tqdm

from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture

from utils.preprocessing import semi_auto_factorize, manual_factorize
from utils.postprocessing import rounding

# Option to display all columns
pd.set_option('display.max_columns', None)


class GMMSyntheticDataGenerator(BaseEstimator):
    def __init__(self):
        self.n_samples = None

        self.original_data = None
        self.data = None

        self.model = None
        self.n_components = None
        self.factorize = []

        self.target = None
        self.target_min = None
        self.target_max = None

        self.original_dtypes = None
        self.dtypes = None
        self.names = None

    def fit(self, data, target, n_components=None, factorize_list=None, make_factorize=False):
        self.original_data = data
        self.target = target
        self.target_min = self.original_data[target].min()
        self.target_max = self.original_data[target].max()

        print('Start building model')

        if factorize_list:
            self.factorize = factorize_list
            self._preprocess(make_factorize)
        else:
            self._preprocess(True)
        if n_components:
            self.n_components = n_components
            self.model = GaussianMixture(n_components=self.n_components,
                                         covariance_type='full').fit(self.original_data)
        else:
            models = []
            bic_scores = []
            n_components_range = range(1, 101)

            for n in tqdm.tqdm(n_components_range):
                model = GaussianMixture(n, covariance_type='full').fit(self.original_data)
                models.append(model)
                bic_scores.append(model.bic(self.original_data))

            self.n_components = n_components_range[np.argmin(bic_scores)]
            print('Number of components:', self.n_components, 'based on Bayes Information criterion')

            self.model = models[n_components_range[np.argmin(bic_scores)]-1]

        print('Model built successfully')

    def _preprocess(self, factorize):
        self.original_dtypes = self.original_data.dtypes.to_dict()
        self.names = self.original_dtypes.keys()
        if factorize:
            self.original_data, self.factorize = semi_auto_factorize(self.original_data)
        self.dtypes = self.original_data.dtypes.to_dict()
        print('Data preproccessed')
        print(self.factorize)

    def _postprocess(self):
        self.data = rounding(self.original_data, self.data, self.dtypes)
        self.data = manual_factorize(self.data, self.factorize, back=False)
        # self.data[self.target] = self.data[self.target].apply(lambda x: 0 if x < 0 else (1 if x > 1 else x))
        self.data[self.target] = np.clip(self.data[self.target], self.target_min, self.target_max)

    def generate(self, num_rows):
        if self.model:
            self.data = pd.DataFrame(self.model.sample(num_rows)[0], columns=self.names)
            self._postprocess()
            return self.data
        else:
            raise TypeError('Model not build')
