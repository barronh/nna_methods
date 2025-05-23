"""
Observationally Corrected EQUATES Ozone
=======================================

Very similar example to plot_correctedozone.py, just for EQUATES instead of for
GEOS-CF. For details about the method[#f1]_, please look at the
detailed description.
This example requires several prerequisite libraries. Install them with the command:

.. code::

    python -m pip install cftime netcdf4 pyrsig pycno git+https://github.com/barronh/nna_methods.git


.. [#f1] Timin, B., Wesson, K., & Thurman, J. (2010). Application of Model and Ambient Data Fusion Techniques to Predict Current and Future Year PM2.5 Concentrations in Unmonitored Areas, Chapter 2.12. In D. G. Steyn & S. T. Rao (Eds.), Air pollution modeling and its application XX. Springer Verlag. dx.doi.org/10.1007/978-90-481-3812-8"""

import os
import time
# unless otherwise specified install: python -m pip install <libname>
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import pyproj
import pyrsig
import pycno
import nna_methods  #  python -m pip install git+https://github.com/barronh/nna_methods.git

njobs = None  # set to 30 to use 30 processes
api = pyrsig.RsigApi(bbox=(-180, 10, 0, 80))
dates = pd.date_range('2019-07-01', '2019-07-01')
outtmpl = 'cmaq.equates.conus.lstaconc.DAILY_O3MAX8_VNA_%F.nc'
for date in dates:
  outpath = date.strftime(outtmpl)
  qpath = 'cmaq.equates.conus.lstaconc.DAILY_O3MAX8'
  aqspath = 'aqs.ozone_8hour_average'
  if os.path.exists(outpath):
    print('cached:', outpath)
    continue
  print('making:', outpath, end='...', flush=True)
  t0 = time.time()
  ds = api.to_ioapi(qpath, bdate=date)
  # aqs.ozone_daily_8hour_maximum doesn't work
  rawdf = api.to_dataframe(aqspath, bdate=date, backend='xdr', unit_keys=False)
  # To work from disk instead:
  # ds = pyrsig.open_ioapi(qpath)
  # rawdf = pd.read_csv(aqspath)
  proj = pyproj.Proj(ds.crs_proj4)
  rawdf['x'], rawdf['y'] = proj(rawdf['LONGITUDE'], rawdf['LATITUDE'])
  df = rawdf.query(
      f'x >= 0 and x < {ds.NCOLS} and y >= 0 and y < {ds.NROWS}'
      'and ozone_8hour_average == ozone_8hour_average'
  ).groupby('STATION').max().copy()
  cidx = df['x'].to_xarray()
  ridx = df['y'].to_xarray()
  df['CMAQ'] = ds['DAILY_O3MAX8'][0, 0].sel(ROW=ridx, COL=cidx, method='nearest').values

  tgtdf = ds[['TSTEP', 'LAY', 'ROW', 'COL']].to_dataframe()
  tgtX = tgtdf.reset_index()[['COL', 'ROW']].values
  t1 = time.time()
  vna = nna_methods.NNA(method='voronoi', k=30)
  vna.fit(df[['x', 'y']], df[['ozone_8hour_average', 'CMAQ']])
  tgtdf[['obs_vna', 'cmaq_vna']] = vna.predict(tgtX, njobs=njobs)
  df[['obs_vna_loo', 'cmaq_vna_loo']] = vna.predict(df[['x', 'y']].values, njobs=njobs, loo=True)
  dfkeys = [
      'x', 'y', 'ozone_8hour_average', 'CMAQ',
      'obs_vna_loo', 'cmaq_vna_loo'
  ]
  tgtds = tgtdf.to_xarray()
  dfds = df[dfkeys].to_xarray()
  for k in dfkeys:
      tgtds['atmon_' + k] = dfds[k]
  # Add derived quantities
  tgtds['eVNA'] = (tgtds['obs_vna'] / tgtds['cmaq_vna']) * ds['DAILY_O3MAX8']
  tgtds['aVNA'] = (tgtds['obs_vna'] - tgtds['cmaq_vna']) + ds['DAILY_O3MAX8']
  keys = list(tgtds.data_vars)
  for k in keys:
    tgtds[k].attrs.update(ds['DAILY_O3MAX8'].attrs)
    tgtds[k].attrs.update(long_name=k.ljust(16), var_desc=k.ljust(80))
  tgtds['atmon_x'].attrs.update(ds.COL.attrs)
  tgtds['atmon_y'].attrs.update(ds.ROW.attrs)
  nv = len(keys)
  tf = ds['TFLAG']
  tgtds['TFLAG'] = tf.dims, tf.sel(VAR=[0] * nv).data, tf.attrs
  tgtds.attrs.update(ds.attrs)
  tgtds.attrs['NVARS'] = nv
  tgtds.attrs['VAR-LIST'] = ''.join([k.ljust(16) for k in keys])
  fdesc = f'nna_methods.{vna} with mod={qpath} and obs={aqspath}'.ljust(60*80)
  tgtds.attrs['FILEDESC'] = fdesc
  tgtds = tgtds[['TFLAG'] + keys]
  outds = tgtds.drop_vars(['TSTEP', 'LAY', 'ROW', 'COL'])
  outds.to_netcdf(outpath, format='NETCDF4_CLASSIC')
  te = time.time()
  print(f'{te - t0:.0f}s ({te - t1:.0f}s)')

# %%
# Plot Raw Model, eVNA, and Difference Maps
# -----------------------------------------
tgtds = xr.open_dataset(outpath)
Z = xr.concat([tgtds['obs_vna'], ds['DAILY_O3MAX8'], tgtds['eVNA'], tgtds['aVNA']], dim='Z')
Z.coords['Z'] = ['OBS', 'CMAQ', 'eVNA', 'aVNA']
cmap = 'viridis'
norm = plt.Normalize(0)
fca = Z.plot(col='Z', col_wrap=2, norm=norm, cmap=cmap)
fca.axs[0, 0].scatter(x=df.x, y=df.y, c=df.ozone_8hour_average, edgecolor='w', norm=norm, cmap=cmap)
pycno.cno(ds.crs_proj4).drawstates(ax=fca.axs);
fca.axs[0, 0].set(ylim=(10, 250), xlim=(10, 420))