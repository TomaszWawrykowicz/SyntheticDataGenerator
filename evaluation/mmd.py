import numpy as np
import pandas as pd
from sklearn import metrics


def mmd_linear(x: pd.DataFrame, y: pd.DataFrame) -> float:
    """
    https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py

    Arguments:
    ---------
        X : features of the studied dataset
        Y : target variable of the studied dataset

    Returns:
    --------
        pcd : the MMD between X and Y
    """
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def mmd_rbf(x: pd.DataFrame, y: pd.DataFrame, gamma=1.0) -> float:
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    xx = metrics.pairwise.rbf_kernel(x, x, gamma)
    yy = metrics.pairwise.rbf_kernel(y, y, gamma)
    xy = metrics.pairwise.rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()
