Nearest Neighbor Methods
========================

Nearest Neighbor methods used for 2-dimensional interpolation. These methods
include two methods of neighbor selection and two methods of weighting. To
access any of the functionality, use the Nearest Neighbor Averaging (`NNA`)
class.

Methodology
-----------

Neighbor Selection:

* `nearest` selects a number (k) of nearest neighbors using euclidian distance.
* `voronoi` selects the Voronoi neighbors from within its k-nearest neighbors.
* `laplace` uses `voronoi` methodology with a special weighting.

Weighting:

* Both `nearest` and `voronoi` use distance power-based weight (`d**power`)
* `laplace` uses the ratio of the voronoi-neighbor edge length do distance between neighbors.


Examples
--------

The example below uses default interpolation methods on California Housing
prices interpolations based solely on longitude, latitude. Not a great test, 
because we know that the model will not fit well, but it demonstrates the
ease of use.

```
import pandas as pd
import numpy as np
from nna_methods import NNA
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing

# last two X featurs are lat and lon
X, y = fetch_california_housing(return_X_y=True)
df = pd.DataFrame(dict(lat=X[:, -2], lon=X[:, -1], y=y, yhat=y*np.nan))
kf = KFold(n_splits=10)

nn = NNA()
for trainidx, testidx in kf.split(X, y):
  nn.fit(X[trainidx, -2:], y[trainidx])
  df.loc[testidx, 'yhat'] = nn.predict(X[testidx, -2:])

statdf = df.describe()
statdf.loc['corr'] = df.corr()['y']
statdf.loc['mb'] = df.subtract(df['y'], axis=0).mean()
statdf.loc['rmse'] = (df.subtract(df['y'], axis=0)**2).mean()**0.5
statdf.loc[['mean', 'std', 'rmse']].round(2)
# Output:
#        lat     lon    y yhat
# mean 35.63 -119.57 2.07 2.09
# std   2.14    2.00 1.15 1.00
# rmse 33.66  121.66 0.00 0.98
```

By default, this exmample uses NNA options `method='nearest'`, `k=10`, and
`power=-2`. That means it selects the 10 nearest neighbors and uses inverse
distance squared weights. You can change that by modifying the line `nn = NNA()`.
For example, `nn = NNA(method='voronoi', k=30, power=-3)` would select Voronoi
neighbors from the nearest 30 points and apply inverse distance cubed weighting.

NNA, eVNA, and aVNA
-------------------

NNA easily implements the method used by the EPA called extended Voronoi
Neighbor Averaging or eVNA. eVNA is used to adjust models to better reflect
observations. For reference, `nn = NNA(method='voronoi', k=30, power=-2)` is
equivalent to the standard options for EPA's `eVNA` method.[1,2] In eVNA, the
obs:model ratio at monitor sites is interpolated to grid cell centers and then
multiplied by the model. The value of `k=3-` generally agrees well softwares
like SMAT-CE and DFT that use an 8 degree radius with a minimum of 20 neighbors.

NNA can also produce a variant of eVNA that I refer to as additive eVNA or
aVNA for short. Instead of interpolating the obs:model ratio, it interpolates
the bias. Then, the bias is subtracted from the model to adjust the model
to better reflect the observations.

References
==========

[1] Timin, Wesson, Thurman, Chapter 2.12 of Air pollution modeling and its
application XX. (Eds. Steyn, D. G., & Rao, S. T.) Dordrecht: Springer Verlag.
May 2009.

[2] Abt, MATS User Guide 2007 or 2010
