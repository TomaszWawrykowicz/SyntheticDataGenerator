import numpy as np
from scipy import special


def kl_div(data_1, data_2):
    # https://github.com/LLNL/SYNDATA/blob/main/utils/performance_metrics.py
    # https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810

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

        # avoid Inf KL
        p[p == 0] = np.finfo(float).eps
        q[q == 0] = np.finfo(float).eps
        p = p / p.sum()
        q = q / q.sum()

        result[col] = special.rel_entr(p, q).sum()

    return result
