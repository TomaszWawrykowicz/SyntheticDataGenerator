import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def graphical_evaluation():
    pass


def hist_plot():
    pass


# wyświetlenie typów zmiennych
# histogramy
# korelacje (mapy ciepła)
# sieć bayesowska dla pokazania zależności?


data = pd.read_csv('heart_uci.csv')
new_data = pd.read_csv('new_data.csv')
data = data[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
             'Oldpeak', 'HeartDisease']]
new_data = new_data[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
                     'Oldpeak', 'HeartDisease']]

# sns.histplot(data, x='Age')
# plt.show()
# sns.histplot(new_data, x='Age')
sns.heatmap(data.corr(), annot=True)
plt.show()

sns.heatmap(new_data.corr(), annot=True)
plt.show()
