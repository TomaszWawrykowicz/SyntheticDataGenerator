import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Option to display all columns
pd.set_option('display.max_columns', None)

df = pd.read_csv('heart_uci.csv')

if __name__ == '__main__':
    print(df.head())
    print(df.nunique())
    print(df.info())
    # print(df.corr())
    # pd.factorize(df['ChestPainType'])
    # df['Sex'] = df['Sex'].factorize()[0]
    for (columnName, columnData) in df.items():
        # print(columnName)
        # print(columnData.dtype)
        if columnData.dtype == object:
            df[columnName] = df[columnName].factorize()[0]

    print(df.head())
    # print(df.info())
    # sns.heatmap(df.corr(), cmap="Blues", annot=True)
    # plt.show()
    # print(df.corr())
    x = df['Sex'].shape
    print(x)

    sns.scatterplot(df, x=np.linspace(0, 918, 918), y=df['MaxHR'])
    plt.show()
