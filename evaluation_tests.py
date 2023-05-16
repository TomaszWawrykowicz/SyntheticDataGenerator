import numpy as np
from scipy.stats import ks_2samp
from scipy.special import rel_entr, kl_div


def ks_test(x, y):
    return ks_2samp(x, y)


def hellinger_distance(x, y):
    # aby to zastosować, należy wektor danych zamienić na rozkład prawdopodobieństwa
    # hist, bin_edges = np.histogram(data, bins='auto', density=True)
    # czyli tak: przyjmuje dwa dataframy x i y -> dla każdej zmiennej przeprowadzam zmiany na hist, a potem badam
    #   jak się różnią od siebie te zmienne

    # jak przeprowadzić badania podobności dla różnych rozmiarów próbek?

    x_hist, bins = np.histogram(x, bins='auto', density=True)
    y_hist = np.histogram(y, bins=len(bins)-1, density=True)[0]
    print(f'Histogram x: {len(x_hist)}'
          f'\n bins - {len(bins)}'
          f'\n y: {len(y_hist)}')
    # vec1, vec2 = dict(x_hist), dict(y_hist)
    sim = np.sqrt(0.5 * ((np.sqrt(x_hist) - np.sqrt(y_hist)) ** 2).sum())
    print(sim)


"""
def hellinger(vec1, vec2):
    https://en.wikipedia.org/wiki/Hellinger_distance
    https://tedboy.github.io/nlps/_modules/gensim/matutils.html#hellinger
    Hellinger distance is a distance metric to quantify the similarity between two probability distributions.
    Distance between distributions will be a number between <0,1>, where 0 is minimum distance (maximum similarity) and 1 is maximum distance (minimum similarity).

    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2): 
        # if it is a bag of words format, instead of converting to dense we use dictionaries to calculate appropriate distance
        vec1, vec2 = dict(vec1), dict(vec2)
        if len(vec2) < len(vec1): 
            vec1, vec2 = vec2, vec1 # swap references so that we iterate over the shorter vector
        sim = numpy.sqrt(0.5*sum((numpy.sqrt(value) - numpy.sqrt(vec2.get(index, 0.0)))**2 for index, value in iteritems(vec1)))
        return sim
    else:
        sim = numpy.sqrt(0.5 * ((numpy.sqrt(vec1) - numpy.sqrt(vec2))**2).sum())
        return sim"""


def kl_div_1(x, y):
    g = False
    if g:
        return rel_entr(x,y)
    return kl_div(x, y)


def kl_div_2(p, q):
    # https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
