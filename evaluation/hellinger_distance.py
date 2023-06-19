import numpy as np


def hellinger_distance(data_1, data_2):
    def __compute_freqs(serie, uniq):
        counts = dict(serie.value_counts())
        p = np.zeros((len(uniq), 1))
        for i, val in enumerate(uniq):
            if val in counts:
                p[i] = counts[val]
        return p / p.sum()

    result = dict()

    for col in data_1.columns:
        uniq_vals = list(data_1[col].unique()) + list(data_2[col].unique())
        uniq_vals = set(uniq_vals)
        p = __compute_freqs(data_1[col], uniq_vals)
        q = __compute_freqs(data_2[col], uniq_vals)

        sim = np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum())

        result[col] = sim
    return result
