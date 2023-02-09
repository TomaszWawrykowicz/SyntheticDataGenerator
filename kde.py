import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns

# Load the data set
df = pd.read_csv('heart_uci.csv')
for (columnName, columnData) in df.items():
    if columnData.dtype == object:
        df[columnName] = df[columnName].factorize()[0]

# Fit a Kernel Density Estimation model to the data
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(df)

# Sample synthetic data from the Kernel Density Estimation model
X_syn_kde = kde.sample(918)
print(X_syn_kde.shape)

# Printing one specific column
column_nb = 2
new_df = []
for row in X_syn_kde:
    # Round for expected int
    apnd = np.round(row[column_nb])

    # Exact value
    # apnd = row[column_nb]

    new_df.append(apnd)

new_df = pd.DataFrame({'col': new_df})
print(new_df)

# Age Sex ChestPainType RestingBP Cholesterol FastingBS RestingECG MaxHR ExerciseAngina Oldpeak ST_Slope HeartDisease
#  0   1    2                3       4           5          6       7        8              9     10           11
# sns.scatterplot(df, x=np.linspace(0, 918, 918), y=df['HeartDisease'].sort_values())
# sns.scatterplot(new_df, x=np.linspace(0, 918, 918), y=new_df['col'].sort_values())
# plt.show()


def get_new_df():
    return new_df
