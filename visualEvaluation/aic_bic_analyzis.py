from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture


def aic_bic(df, n_min, n_max=None):
    # Define the range of values for n_components
    if n_max:
        n_components_range = range(n_min, n_max)
    else:
        n_components_range = n_min

    # Initialize lists to store the AIC and BIC values
    aic_values = []
    bic_values = []

    # Fit the GMM for each value of n_components
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=500, n_init=3, verbose=1)
        gmm.fit(df)
        aic_values.append(gmm.aic(df))
        bic_values.append(gmm.bic(df))

    # Plot the AIC and BIC values
    plt.plot(n_components_range, aic_values, label='AIC')
    plt.plot(n_components_range, bic_values, label='BIC')
    plt.legend()
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.show()
