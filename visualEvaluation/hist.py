import math
import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.preprocessing import manual_factorize, semi_auto_factorize


def one_image_hist(data_1, data_2, labels, density=False, title=None, x=None, y=None):
    data_1, fact = semi_auto_factorize(data_1)
    data_2 = manual_factorize(data_2, fact, clip=False, input_fact=True)

    fig = plt.figure(figsize=(25.6, 14.4), dpi=100)
    if title is None:
        title = 'Chart'
    if x is None and y is None:
        x = 2
        y = math.ceil(data_1.shape[1] / 2)

    for idx, columnName in enumerate(data_1.columns):
        fig.add_subplot(x, y, idx + 1)
        plt.hist([data_1[columnName], data_2[columnName]], color=['r', 'b'], alpha=0.8, density=density)

        for col in fact:
            if col[0] == columnName:
                plt.xticks(list(col[1].keys()), list(col[1].values()))
        plt.title(f'{columnName}', fontsize=20)
    fig.suptitle(title, fontsize=32)
    fig.legend(labels=labels, loc='upper right',
               bbox_to_anchor=(0.95, 0.99), fontsize=16)
    plt.savefig(title + '_' + str(random.randint(10000, 99999)) + '.png')
    plt.show()


def hist(data_1, data_2, density=False):
    data_1, fact = semi_auto_factorize(data_1)
    data_2 = manual_factorize(data_2, fact, clip=False, input_fact=True)

    for idx, columnName in enumerate(data_1.columns):
        plt.hist([data_1[columnName], data_2[columnName]], color=['r', 'b'], alpha=0.8,
                 label=['Original', 'Synthetic'], density=density)
        plt.legend()
        for col in fact:
            if col[0] == columnName:
                plt.xticks(list(col[1].keys()), list(col[1].values()))
        plt.title(f'Hist of {columnName}')
        plt.show()

# def pair_plot(data):
#     sns.boxplot(data)
#     plt.show()
