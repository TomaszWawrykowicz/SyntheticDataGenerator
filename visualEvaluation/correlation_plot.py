import random

import seaborn as sns
from matplotlib import pyplot as plt


def correlation_plot_one_dataset(data, title='chart', method='pearson'):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    sns.heatmap(data.corr(method=method), annot=True, cmap='coolwarm', vmax=1, vmin=-1)
    plt.title(title, fontsize=32)
    plt.savefig(title + '_' + str(random.randint(10000, 99999)) + '.png')
    plt.show()


def correlation_plot_two_datasets(data_1, data_2, method='pearson'):
    data = data_1.corrwith(data_2, method=method)
    sns.barplot(x=data.index, y=data.values, color='#C41230')
    plt.xticks(rotation=30)
    plt.title(f'{method} correlation for data attributes of datasets')
    plt.show()
