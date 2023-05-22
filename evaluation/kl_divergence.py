import numpy as np


def kl_div(data_1, data_2):
    # https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
    results = []
    for col in data_1.columns:
        x_hist, bins = np.histogram(data_1[col], bins='auto', density=True)
        y_hist = np.histogram(data_2[col], bins=len(bins) - 1, density=True)[0]

        x_hist /= np.sum(x_hist)
        y_hist /= np.sum(y_hist)

        # Adding epsilon for probabilities equal to 0
        x_hist += 1e-8
        y_hist += 1e-8

        val = np.sum(x_hist * np.log(x_hist / y_hist))
        results.append({col: val})
    return results
