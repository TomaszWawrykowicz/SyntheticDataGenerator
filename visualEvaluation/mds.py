import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import MDS


def mds_method(dataframe):
    model2d = MDS(n_components=2)
    x_trans = model2d.fit_transform(dataframe)
    mds_df = pd.DataFrame(data=x_trans, columns=['Standard X', 'Standard Y'])

    plt.clf()
    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(data=mds_df, x='Standard X', y='Standard Y', s=50)
    ax.set_title(
        f'The new shape of X: {x_trans.shape}, No. of Iterations: {model2d.n_iter_}, Stress: {model2d.stress_}',
        fontsize=20)
    plt.show()
