import numpy as np
import pandas as pd
from evaluation_tests import hellinger_distance

s = pd.read_csv('heart_uci.csv')
df = pd.read_csv('new_data.csv')
print(df.head())
# print(s['Age'])

# hellinger_distance(s['Age'], df['Age'])

# print(((np.sqrt([1, 2, 3, 4]) - np.sqrt([2, 2, 3, 3, 4, 5, 6])) ** 2).sum())

new_data = df[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
               'Oldpeak', 'HeartDisease']]

old_data = s[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
              'Oldpeak', 'HeartDisease']]

from visualEvaluation.correlation_plot import correlation_plot_one_sample, correlation_plot_two_samples
print(old_data.shape)
print(new_data.shape)
correlation_plot_two_samples(new_data, old_data)
