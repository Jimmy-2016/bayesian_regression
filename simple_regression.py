# Param Recovery for Baysian Regression

import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az


# Simulate some data (time series features X and target Y)
n_samples = 1000
n_features = 4
np.random.seed(42)
X = np.random.randn(n_samples, n_features)
beta_true = np.array([0.5, -1.0, 0.3, 2])  # True coefficients
Y = np.dot(X, beta_true) + np.random.normal(0, 0.5, size=n_samples)  # Linear model with noise

with pm.Model() as model:
    # Priors
    beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)
    intercept = pm.Normal('intercept', mu=0, sigma=10)

    # Prior for error term
    sigma = pm.HalfNormal('sigma', sigma=1)

    # sigma for likelihood, this can approximate Geoffry Prior
    likelihood_sigma = pm.HalfCauchy('likelihood_sigma', beta=1)

    # Likelihood (normal distribution for observed Y)
    mu = intercept + pm.math.dot(X, beta)
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=likelihood_sigma, observed=Y)

    # Sample from the posterior using MCMC
    trace = pm.sample(100, return_inferencedata=True)


## inference
X_new = np.random.randn(50, n_features)  # 10 new samples with same number of features
y_true = np.dot(X_new, beta_true)
with pm.Model() as model_new:
    beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    likelihood_sigma = pm.HalfNormal('likelihood_sigma', sigma=1)

    # likelihood for the new data
    mu_new = intercept + pm.math.dot(X_new, beta)
    Y_new = pm.Normal('Y_new', mu=mu_new, sigma=likelihood_sigma)

    # Sample predictions
    posterior_predictive_new = pm.sample_posterior_predictive(trace, var_names=["Y_new"], random_seed=42)



y_pred = posterior_predictive_new.posterior_predictive['Y_new'][0, -1]
plt.figure()
plt.plot(y_pred, 'b', label='predict')
plt.plot(y_true, 'r', label='true')
plt.legend()
# Posterior summary
print(az.summary(trace, round_to=2))

# Plot posterior distributions
az.plot_posterior(trace)

# parameter updates over time
az.plot_trace(trace)

# final posterior of params
az.plot_posterior(trace)

plt.show()