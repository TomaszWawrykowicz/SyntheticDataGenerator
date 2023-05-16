import seaborn as sns
from matplotlib import pyplot as plt


def correlation_plot_one_sample(data, method='pearson'):
    sns.heatmap(data.corr(), annot=True)
    plt.show()


def correlation_plot_two_samples(data_1, data_2, method='pearson'):
    data = data_1.corrwith(data_2, method=method)
    print('data geggege')
    print(data)
    print('to nie może być chyba połączone z corrwith - tam zwraca arrajkę z różnicami')
    sns.heatmap(data, annot=True)
    plt.show()
