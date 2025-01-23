Nearest Neighbor Methods
========================

Nearest Neighbor methods used for 2-dimensional interpolation. These methods
include two methods of neighbor selection and two methods of weighting. To
access any of the functionality, use the Nearest Neighbor Averaging (`NNA`)
class.

Methodology
-----------

### NNA

Neighbor Selection:

* `nearest` selects a number (k) of nearest neighbors using euclidian distance.
* `voronoi` selects the Voronoi neighbors from within its k-nearest neighbors.
* `laplace` uses `voronoi` methodology with a special weighting.

Weighting:

* Both `nearest` and `voronoi` use distance power-based weight (`d**power`)
* `laplace` uses the ratio of the voronoi-neighbor edge length do distance between neighbors.

### GMOS

An additional option has been implemented to use iterative radius searches and
subsequent smoothing. This method is based on work done at NOAA/NWS by Glahn (2009)


Examples
--------

The example below uses default interpolation methods on California Housing
prices interpolations based solely on interpolating sparse data. We do not
expect the model to fit well and the correlation is around 0.6. The intent
is not to make a great model of housing, but to show the ease of use.

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
multiplied by the model. The value of `k=30` generally agrees well softwares
like SMAT-CE and DFT that use an 8 degree radius with a minimum of 20 neighbors.

NNA can also produce a variant of eVNA that I refer to as additive eVNA or
aVNA for short. Instead of interpolating the obs:model ratio, it interpolates
the bias. Then, the bias is subtracted from the model to adjust the model
to better reflect the observations.

NNA, eIDW, and aIDW
-------------------

The eVNA and aVNA approaches described above can be implemented with nearest
instead of vornoi neighbors in about 2% of the walltime. Although it is fast,
IDW is subject to artifacts due to the spatially biased observation network
that eVNA was designed to address. In particular, the cluster observations
around an urban area can overwhelm observations that are closer, but isolated.
This can be overcome increasing the magnitude of the power (e.g., -2 to -5).
For some applications, that might be fine.

DelaunayInterp, eDNA, and aDNA
------------------------------

The eVNA and aVNA approaches described above can be implemented with Delaunay
triangulation in about 0.25% of the walltime. This is very fast, but uses only
the three points of the simplex that each prediction coordinate is within. This
has the benefit of now allowing an isolated point to be overwhelmed by a
cluster, but has ridges associated with the boundaries of interpolation within
tesselation (triangle) borders.

GMOS, eGMOS, and aGMOS
----------------------

GMOS is an implementation of the gridding of model output statistics (GMOS).
GMOS was coined by Glahn et al. (2009) and a clear implementation is available
in Glahn, Im and Wagner (2012). This technique is used by Djalalova (2015) and
subsequently by NOAA for various "adjustments" to their forecasts. Below is my
broad-brush understanding:

    i \in AirNow stations
    y_i = NAQFC_i
    o_{a,i}, y_{a,i} = Analog(y_i, t_i, ws_i, wd_i, sr_i)
    k_i = KalmanFilter(o_{a,i})
    c_i = y_i - mean(y_{a,i}) 
    b_i = y_i - (k_i + c_i)
    y'_x = y_x - GMOS(b_i)

If we are interested in applying a known bias, unlike Djalalova (2015), then
b_i does not need to be predicted. In that case, bias ($b_i$) can be estimated
from the difference between the model ($NAQFC_i$) and the observed value at the
station ($o_i$). Then, GMOS can be used in place of VNA for an aVNA-like
product. Or, the ratio could be used to create an eVNA-like product. The GMOS
method is about 25% faster than the VNA equivalent. The results are more
similar to voronoi than IDW. This likely due to the concentric circles reducing
in successive iterations. Thus, reducing the influence of non-voronoi
neighbors.

Note: The GMOS implemented here does not have elevation or land/ocean
awareness. Nor does it implement iterative QC as described by Glahn (2009).


References
==========

[1] Timin, Wesson, Thurman, Chapter 2.12 of Air pollution modeling and its
application XX. (Eds. Steyn, D. G., & Rao, S. T.) Dordrecht: Springer Verlag.
May 2009.

[2] Abt, MATS User Guide 2007 or 2010

[3] Glahn, Gilbert, Cosgrove, Ruth, and Sheets: The Gridding of MOS, Weather
and Forecasting, 24, 520-529, https://doi.org/10.1175/2008WAF2007080.1, 2009.

[4] Glahn, B., Im, J. S., and Wagner, G.: 8.4 Objective Analysis of MOS
Forecasts and Observations in Sparse Data Regions, 2012.

[5] Djalalova, I., Delle Monache, L., and Wilczak, J.: PM2.5 analog forecast
and Kalman filter post-processing for the Community Multiscale Air Quality (CMAQ) model,
Atmospheric Environment, 108, 76-87, https://doi.org/10.1016/j.atmosenv.2015.02.021, 2015.

