import pandas as pd
from numpy import linalg


def PCD(x: pd.DataFrame, x_ref: pd.DataFrame) -> float:
    # https://github.com/antorguez95/synthetic_data_generation_framework/blob/master/sdg_utils.py#L40
    """ This function computes and return the Pairwise Correlation
        Difference (PCD). PCD formulation can be found in [1].

        Arguments:
        ---------
        X : features of the studied dataset
        X_ref : features of the reference dataset

        Returns:
        --------
        pcd : pcd value

        References:
        -----------
        [1] Generation and evaluation of ______

        """

    # Correlation difference
    dif_corr = x.corr().fillna(0) - x_ref.corr().fillna(0)  # if nan in corelation

    # Frobenius norm of calculated difference
    pcd = linalg.norm(dif_corr, ord='fro')

    return pcd
