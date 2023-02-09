import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture

from kde import get_new_df

n_components = 16

# Load the data set
df = pd.read_csv('heart_uci.csv')
for (columnName, columnData) in df.items():
    if columnData.dtype == object:
        df[columnName] = df[columnName].factorize()[0]


# Build model
gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=500, n_init=3, verbose=1)
gmm.fit(df)

# Sample synthetic data from the Gaussian Mixture model
gmm_samples = gmm.sample(918)[0]
print(gmm_samples)

# Printing one specific column
column_nb = 2
new_df = []
for row in gmm_samples:
    # Round for expected int
    apnd = np.round(row[column_nb])

    # Exact value
    # apnd = row[column_nb]

    new_df.append(apnd)

new_df = pd.DataFrame({'col': new_df})
print(new_df)

# Age Sex ChestPainType RestingBP Cholesterol FastingBS RestingECG MaxHR ExerciseAngina Oldpeak ST_Slope HeartDisease
#  0   1    2                3       4           5          6       7        8              9     10           11
sns.scatterplot(df, x=np.linspace(0, 918, 918), y=df['ChestPainType'].sort_values())
sns.scatterplot(new_df, x=np.linspace(0, 918, 918), y=new_df['col'].sort_values())

kde_sample = get_new_df()
sns.scatterplot(kde_sample, x=np.linspace(0, 918, 918), y=kde_sample['col'].sort_values())
plt.show()
