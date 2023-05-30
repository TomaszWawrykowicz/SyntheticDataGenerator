import pandas as pd

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
        self.factorize = []

        self.original_dtypes = None
        self.dtypes = None
        self.names = None

    def fit(self, data, n_components):
        self.original_data = data
        self._preprocess()

        self.model = GaussianMixture(n_components=n_components,
                                     covariance_type='full').fit(self.original_data)
        print('Model built successfully')

    def _preprocess(self):
        self.original_dtypes = self.original_data.dtypes.to_dict()
        self.names = self.original_dtypes.keys()
        self.original_data, self.factorize = semi_auto_factorize(self.original_data)
        self.dtypes = self.original_data.dtypes.to_dict()
        print('Data preproccessed')
        print(self.factorize)

    def _postprocess(self):
        self.data = rounding(self.original_data, self.data, self.dtypes)
        self.data = manual_factorize(self.data, self.factorize, back=True)

    def generate(self, num_rows):
        if self.model:
            self.data = pd.DataFrame(self.model.sample(num_rows)[0], columns=self.names)
            self._postprocess()

            # print(self.original_data.info())
            # print(self.data.info())
            # print(self.original_data.describe())
            # print(self.data.describe())
            return self.data
        else:
            raise TypeError('Model not build')
