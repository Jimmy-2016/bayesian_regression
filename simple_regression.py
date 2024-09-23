import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az


# Simulate some data (for example, time series features X and target Y)
n_samples = 100
n_features = 3
np.random.seed(42)
X = np.random.randn(n_samples, n_features)
beta_true = np.array([0.5, -1.0, 0.3])  # True coefficients
Y = np.dot(X, beta_true) + np.random.normal(0, 0.5, size=n_samples)  # Linear model with noise

# Define Bayesian model
with pm.Model() as model:
    # Priors for coefficients
    beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)
    intercept = pm.Normal('intercept', mu=0, sigma=10)

    # Prior for error term
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Likelihood (normal distribution for observed Y)
    mu = intercept + pm.math.dot(X, beta)
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

    # Sample from the posterior using MCMC
    trace = pm.sample(2, return_inferencedata=True)



# Posterior summary
print(az.summary(trace, round_to=2))

# Plot posterior distributions
az.plot_posterior(trace)

# Trace plot to monitor parameter updates over time
az.plot_trace(trace)

# Posterior plot to visualize the final distributions of parameters
az.plot_posterior(trace)

# log_likelihood_vals.plot()

plt.show()