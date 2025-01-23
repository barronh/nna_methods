"""
Observationally Corrected Model Ozone
=====================================

This example shows how to use nna_methods to perform a model bias correction.
In plain language, we correct the model (:math:`m`) based on observations (:math:`o`) by
interpolating the relative error from sites (:math:`r=o/m`). The interpolation
uses the Voronoi neighbors (:math:`n`) and inverse distance squared between each
observation and the prediction site as the weight (:math:`w_n`). This approach is
known as Extended Voronoi Neighbor Averaging (eVNA, [#f1]_). The equation is
summarized as:

.. math::

    eVNA_x = m_x r_x
    
    r_x =sum(w_n r_n)/sum(w_n)

    w_n = d_n^{-2}

    d_n = ((x_n - x_x)^2 + (y_n - y_x)^2)^{0.5}


So, we need a model and observations. For this example:

- :math:`m`: NASA Goddard Modeling and Assimmilation Office routine predictions
  by the GEOS Composition Forecast system (GEOS-CF).
- :math:`m`: AirNow Ozone from EPAs Remote Sensing Information Gateway RSIG
- :math:`x` and :math:`y`: for this example, we'll use lon/lat to calculate
  distances.

.. [#f1] Timin, B., Wesson, K., & Thurman, J. (2010). Application of Model and Ambient Data Fusion Techniques to Predict Current and Future Year PM2.5 Concentrations in Unmonitored Areas, Chapter 2.12. In D. G. Steyn & S. T. Rao (Eds.), Air pollution modeling and its application XX. Springer Verlag. dx.doi.org/10.1007/978-90-481-3812-8"""

# Install library requirements if needed.
#%pip install cftime netcdf4 pyrsig git+https://github.com/barronh/nna_methods.git
import pandas as pd
import numpy as np
import cftime
import xarray as xr
import nna_methods
import pyrsig
import pycno
import matplotlib.pyplot as plt

# %%
# Define Domain of interest
# ----------------------------------

bbox = (-130, 20, -55, 55)     # Define the bounding box of the analysis

date = pd.to_datetime('2024-07-01T17:30Z')  # midpoint of the hour of interest


# %%
# Get ozone observations from airnow
# ----------------------------------

bdate = date.strftime('%Y-%m-%dT%H:00:00Z')  # RSIG time bounds
edate = date.strftime('%Y-%m-%dT%H:59:59Z')
api = pyrsig.RsigApi(bbox=bbox, bdate=bdate, edate=edate)
obsdf = api.to_dataframe(
    'airnow.ozone', unit_keys=False, parse_dates=True
).query(
    'ozone == ozone'  # require valid inputs
)

# %%
# Get ozone model from GEOS-CF
# ----------------------------

cfpath = (
    'https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/assim/'
    + 'aqc_tavg_1hr_g1440x721_v1'
)
cff = xr.open_dataset(cfpath, decode_cf=False)  # save time by note decoding
time = cftime.date2num(date, cff['time'].units, calendar='standard')
o3 = cff['o3'].sel(
    lev=72, time=time, method='nearest'
).sel(
    lat=slice(*bbox[1::2]), lon=slice(*bbox[::2])
) * 1e9
o3.attrs.update(long_name='geoscf', units='ppb')
o3.coords['time'] = date
o3.name = 'geoscf'

# %%
# Pair model with observations
# ----------------------------

latidx = obsdf['LATITUDE'].to_xarray()
lonidx = obsdf['LONGITUDE'].to_xarray()
obsdf['geoscf'] = o3.sel(lat=latidx, lon=lonidx, method='nearest')
obsdf['ratio'] = obsdf.eval('ozone / geoscf')

# %%
# Use Voronoi Neighbor Interpoaltion
# ----------------------------------

xkeys = ['LONGITUDE', 'LATITUDE']
ykeys = ['ozone', 'geoscf', 'ratio']
nn = nna_methods.NNA(k=30, power=-2, method='voronoi')
nn.fit(obsdf[xkeys], obsdf[ykeys])

tgtdf = o3.to_dataframe()
tgtX = tgtdf[[]].reset_index()[['lon', 'lat']]
tgtdf.loc[:, ['vna_' + k for k in ykeys]] = nn.predict(tgtX)
tgtds = tgtdf.to_xarray()
# add eVNA
tgtds['geoscf'].attrs.update(long_name='GEOSCF Ozone', units='ppb')
tgtds['vna_ozone'].attrs.update(long_name='VNA(AirNow Ozone)', units='ppb')
tgtds['vna_geoscf'].attrs.update(long_name='VNA(GEOSCF Ozone)', units='ppb')
tgtds['vna_ratio'].attrs.update(long_name='VNA(AirNow/GEOSCF)', units='1')
tgtds['evna'] = tgtds.eval('geoscf * vna_ratio')
tgtds['evna'].attrs.update(long_name='eVNA Ozone', units='ppb')
tgtds.attrs.update(
    title='GEOS-CF eVNA Ozone', institution='your affiliation',
    source=f'nna_methods {nna_methods.__version__}',
    references=(
        'AirNow https://docs.airnowapi.org/;'
        'Keller et al. doi:10.1029/2020MS002413;'
        'Timin et al. doi:10.1007/978-90-481-3812-8;'
    ),
    history=f'Created {pd.to_datetime("now").strftime("%FT%H:%M:%S%z")}',
    comment=(
        'Ozone observations from AirNow;'
        'Model ozone from GEOS-CF https://fluid.nccs.nasa.gov/cf/;'
        'nna_methods.NNA(k=30, method="voronoi", power=-2);'
    ),
    
)
# optionally, save result to disk
# tgtds.to_netcdf(f'geoscf_evna_{date:%Y-%m-%dT%H%M}.nc')

# sample model output at observations (comparing centroid to sites)
atobsds = tgtds.sel(lat=latidx, lon=lonidx, method='nearest')
obsdf['vna_ozone'] = atobsds['vna_ozone']
obsdf['vna_geoscf'] = atobsds['vna_geoscf']
obsdf['evna'] = atobsds['evna']
# optionally, save result to disk
# obsdf.to_csv(f'geoscf_vna_atmon_{date:%Y-%m-%dT%H%M}.csv')

# %%
# Calculate Model Performance Statistics
# --------------------------------------

okey = 'ozone'
ykeys = ['ozone', 'geoscf', 'evna']
err = obsdf[ykeys].subtract(obsdf[okey], axis=0)
statdf = obsdf[ykeys].describe()
statdf.index.name = 'metric'
statdf.loc['r'] = obsdf[ykeys].corr().loc[okey]
omean = statdf.loc['mean', okey]
statdf.loc['mb'] = statdf.loc['mean'].subtract(omean)
statdf.loc['nmb_pct'] = statdf.loc['mb'].divide(omean) * 100
statdf.loc['rmse'] = (err**2).mean()**.5
# optionally, save out
# statdf.to_csv(f'geoscf_vna_atmon_mpe_{date:%Y-%m-%dT%H%M}.csv')

# %%
# Plot Evaluation with Model Performance Statistics
# -------------------------------------------------

gskw = dict(left=0.05, right=0.98)
fig, axx = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw=gskw)
axx[1].set_axis_off()
tdf = statdf.round(2).reset_index()
axx[1].table(cellText=tdf.values, colLabels=tdf.columns, loc='center')
ax = axx[0]
scat = obsdf.plot.scatter
scat(x=okey, y='geoscf', label='GEOS-CF', color='grey', ax=ax)
scat(x=okey, y='evna', label='eVNA', color='k', marker='+', ax=ax)
ax.axline((0, 0), slope=1, label='1:1')
_ = ax.legend()

# %%
# Plot Raw Model, eVNA, and Difference Maps
# -----------------------------------------

fig, axx = plt.subplots(1, 3, figsize=(13.5, 4), gridspec_kw=gskw)
norm = plt.Normalize()
tgtds['geoscf'].plot(ax=axx[0], norm=norm, cmap='viridis')
axx[0].set(title='GEOS-CF')
tgtds['evna'].plot(ax=axx[1], norm=norm, cmap='viridis')
axx[1].set(title='eVNA')
Z = tgtds.eval('evna - geoscf')
Z.attrs.update(long_name='eVNA - CF', units='ppb')
Z.plot(ax=axx[2], cmap='seismic')
axx[2].set(title='Diff = eVNA - CF')
pycno.cno().drawstates(ax=axx)