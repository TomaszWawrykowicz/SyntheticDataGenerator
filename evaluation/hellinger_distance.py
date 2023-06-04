import numpy as np


def hellinger_distance(data_1, data_2):
    result = dict()
    for col in data_1.columns:
        # Transform given column to probability list
        x_hist, bins = np.histogram(data_1[col], bins='auto', density=True)
        y_hist = np.histogram(data_2[col], bins=len(bins) - 1, density=True)[0]

        # Compute Hellinger Distance value
        sim = np.sqrt(0.5 * ((np.sqrt(x_hist) - np.sqrt(y_hist)) ** 2).sum())

        result[col] = sim
    return result
