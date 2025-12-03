# This script requires:
# 1. A fitted model (Gaussian Naive Bayes in this case) with the following attributes:
#    - X_train: Training feature data (Pandas DataFrame or Numpy array).
#    - y_train: Training target data (Pandas Series or Numpy array).
#    - X_test: Test feature data (Pandas DataFrame or Numpy array).
# 2. The model should have been trained with the GaussianNB class from sklearn.
# 3. The script calculates feature-wise contributions to the prediction of a single test sample using Gaussian log-likelihood and class priors.

# Find which feature has contributed how much
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Fit model
model = GaussianNB()
model.fit(X_train, y_train)

# Pick one test sample
x = X_test.iloc[0].values     # row vector
classes = model.classes_

means = model.theta_          # (n_classes, n_features)
variances = model.var_        # (n_classes, n_features)
priors = model.class_prior_   # (n_classes,)

contributions = {}

for idx, cls in enumerate(classes):
    mu = means[idx]
    var = variances[idx]

    # Gaussian log-likelihood for each feature
    loglik = -0.5 * np.log(2 * np.pi * var) - ((x - mu) ** 2) / (2 * var)

    total_score = np.log(priors[idx]) + loglik.sum()

    contributions[cls] = {
        "feature_contrib": loglik,
        "total_score": total_score
    }

for cls, info in contributions.items():
    print(f"\nClass: {cls}")
    print("Feature contributions:")
    for f, val in zip(X_train.columns, info["feature_contrib"]):
        print(f"  {f}: {val:.4f}")
    print("Total class score:", info["total_score"])

'''
Explanation
pclass (-0.7987)
This passenger’s class (1/2/3) somewhat decreases the probability of class 0
→ Not strongly negative, so a mild effect.

gender (0.0293)
Slightly positive, meaning the gender fits class 0 better than class 1.
→ Typically males (encoded as 1) slightly support not surviving.

age_scaled (-1.0015)
The passenger’s age reduces the likelihood of class 0 moderately.

Total:

-2.2544 → less negative than class 1 → class 0 is more likely.


pclass (-1.5218)
This passenger’s pclass is much worse for predicting survival.
Example: a 3rd-class passenger heavily reduces survival probability.

gender (-1.2549)
Gender strongly argues against survival.
Likely the person is male → men had much lower survival rates.

age_scaled (-1.0223)
Age also slightly reduces survival likelihood.
'''
