import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_method(dataframe, n_components=None):
    pca = PCA(n_components=n_components)
    x = StandardScaler().fit_transform(dataframe)
    principal_components = pca.fit_transform(x)
    if n_components == 2:
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])
        plt.clf()
        plt.figure(figsize=(10, 8))

        ax = sns.scatterplot(data=principal_df, x='principal component 1', y='principal component 2',
                             s=60)

        ax.set_xlabel(
            f'Principal Component 1: '
            f'{round(pca.explained_variance_ratio_[0] * 100, 2)}% \n of explained variance ratio in data',
            fontsize=15)
        ax.set_ylabel(
            f'Principal Component 2: '
            f'{round(pca.explained_variance_ratio_[1] * 100, 2)}% \n of explained variance ratio in data',
            fontsize=15)
        ax.set_title(
            f'From {dataframe.shape[1]} to 2 Component PCA: '
            f'{round(np.cumsum(pca.explained_variance_ratio_ * 100)[1], 2)}% of explained variance ratio'
            f' in data',
            fontsize=20)
        plt.show()
