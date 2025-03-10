"""
Interpolated California Housing Price
=====================================

The example below uses default interpolation methods on California Housing
prices interpolations based solely on interpolating sparse data. We do not
expect the model to fit well and the correlation is around 0.6.

This example requires several libraries that can be installed with the command:

.. code::

    python -m pip install git+https://github.com/barronh/nna_methods.git

"""

# %%
# Import Libraries
# ----------------
import pandas as pd
import numpy as np
from nna_methods import NNA
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing

# %%
# Get Testing Data
# ----------------
# last two X featurs are lat and lon
X, y = fetch_california_housing(return_X_y=True)
df = pd.DataFrame(dict(lat=X[:, -2], lon=X[:, -1], y=y, yhat=y*np.nan))

# %%
# Perform Cross Validation
# ------------------------
# Using 10-fold as is a common practice

kf = KFold(n_splits=10)
nn = NNA()
for trainidx, testidx in kf.split(X, y):
  nn.fit(X[trainidx, -2:], y[trainidx])
  df.loc[testidx, 'yhat'] = nn.predict(X[testidx, -2:])

# %%
# Output Model Performance Statistics
# -----------------------------------
# Using 10-fold as is a common practice

statdf = df.describe()
statdf.loc['r'] = df.corr()['y']
statdf.loc['mb'] = df.subtract(df['y'], axis=0).mean()
statdf.loc['rmse'] = (df.subtract(df['y'], axis=0)**2).mean()**0.5
print(statdf.loc[['mean', 'std', 'rmse', 'r']].round(2).to_markdown())
# Output:
# |      |   lat |     lon |    y |   yhat |
# |:-----|------:|--------:|-----:|-------:|
# | mean | 35.63 | -119.57 | 2.07 |   2.09 |
# | std  |  2.14 |    2    | 1.15 |   1    |
# | rmse | 33.66 |  121.66 | 0    |   0.98 |
# | r    | -0.14 |   -0.05 | 1    |   0.59 |