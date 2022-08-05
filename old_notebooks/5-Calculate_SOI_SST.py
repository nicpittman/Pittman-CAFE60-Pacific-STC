# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
#
# TNI computation: 
#
# (a) Compute area averaged total SST from Niño 1+2 region; 
#
# (b) Compute area averaged total SST from Niño 4 region; 
#
# (c) Compute monthly climatologies (e.g., 1950-1979) for area averaged total SST from Niño 1+2 region, and Niño 4 region, and subtract climatologies from area averaged total SST time series to obtain anomalies; 
#
# (d) Normalize each time series of anomalies by their respective standard deviations over the climatological period; 
#
# (e) Define the raw TNI as Niño 1+2 normalized anomalies minus Niño 4 normalized anomalies; 
#
# (f) Smooth the raw TNI with a 5-month running mean; 
#
# (g) Normalize the smoothed TNI by its standard deviation over the climatological period.

# Niño 1+2 (0-10S, 90W-80W):  The Niño 1+2 region is the smallest and eastern-most of the Niño SST regions, and corresponds with the region of coastal South America where El Niño was first recognized by the local populations.  This index tends to have the largest variance of the Niño SST indices.
#
# Niño 3 (5N-5S, 150W-90W):  This region was once the primary focus for monitoring and predicting El Niño, but researchers later learned that the key region for coupled ocean-atmosphere interactions for ENSO lies further west (Trenberth, 1997).  Hence, the Niño 3.4 and ONI became favored for defining El Niño and La Niña events.
#
# Niño 3.4 (5N-5S, 170W-120W):  The  Niño 3.4 anomalies may be thought of as representing the average equatorial SSTs across the Pacific from about the dateline to the South American coast.  The Niño 3.4 index typically uses a 5-month running mean, and El Niño or La  Niña events are defined when the  Niño 3.4 SSTs exceed +/- 0.4C for a period of six months or more.
#
# ONI (5N-5S, 170W-120W): The ONI uses the same region as the Niño 3.4 index.  The ONI uses a 3-month running mean, and to be classified as a full-fledged El Niño or La Niña, the anomalies must exceed +0.5C or -0.5C for at least five consecutive months.  This is the operational definition used by NOAA.
#
# Niño 4 (5N-5S, 160E-150W): The  Niño 4 index captures SST anomalies in the central equatorial Pacific.  This region tends to have less variance than the other Niño regions.

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def convert_lon(lon):
    ans=180-(lon-180)
    return ans


convert_lon(150)


def cut_sst():
    physics_month= xr.open_zarr('/OSM/CBR/OA_DCFP/data/model_output/CAFE/data_assimilation/d60-zarr/ocean_month.zarr',consolidated=True)
    physics_month['xt_ocean']=physics_month['xt_ocean']+360
    sst=physics_month.sst.sel(xt_ocean=slice(120,290),yt_ocean=slice(-20,20))
    sst['time']=np.array(sst.indexes['time'].to_datetimeindex(), dtype='datetime64[M]')
    sst=sst.sel(time=slice(np.datetime64('1982-01-01'),np.datetime64('2020-01-01')))
    sst=sst.rename({'xt_ocean':'lon','yt_ocean':'lat'})
    sst.to_netcdf('/scratch1/pit071/CAFE60/sst.nc')
    
    #Save SOI index.
    nino34=sst.sel(lat=slice(-5,5),lon=slice(190,240))
    soi=(nino34.mean(dim=['time','lat','lon']))-nino34.mean(dim=['lat','lon'])
    soi.sst.to_netcdf('/scratch1/pit071/CAFE60/soi.nc')
    #return sst
   


sst=cut_sst()

sst=xr.open_dataset('/scratch1/pit071/CAFE60/sst.nc')

sst.sst.mean(dim=['time','ensemble']).plot()

nino12=sst.sel(lat=slice(-10,0),lon=slice(270,280))
nino3=sst.sel(lat=slice(-5,5),lon=slice(210,270))
nino34=sst.sel(lat=slice(-5,5),lon=slice(190,240))
ONI=sst.sel(lat=slice(-5,5),lon=slice(190,240))
nino4=sst.sel(lat=slice(-5,5),lon=slice(160,210))

nino34



mean_soi=soi.sst.mean(dim='ensemble')
mean_soi.plot()

mean_soi

# (a) Compute area averaged total SST from Niño 1+2 region;
#
# (b) Compute area averaged total SST from Niño 4 region;
#
# (c) Compute monthly climatologies (e.g., 1950-1979) for area averaged total SST from Niño 1+2 region, and Niño 4 region, and subtract climatologies from area averaged total SST time series to obtain anomalies;
#
# (d) Normalize each time series of anomalies by their respective standard deviations over the climatological period;
#
# (e) Define the raw TNI as Niño 1+2 normalized anomalies minus Niño 4 normalized anomalies;
#
# (f) Smooth the raw TNI with a 5-month running mean;
#
# (g) Normalize the smoothed TNI by its standard deviation over the climatological period.

# +
a=nino12.mean(dim=['lat','lon']).sst
b=nino4.mean(dim=['lat','lon']).sst

c_a_clim=a.groupby("time.month").mean("time")
c_a=a.groupby('time.month')
c_a1=c_a_clim-c_a

c_b_clim=b.groupby("time.month").mean("time")
c_b=b.groupby('time.month')
c_b1=c_b_clim-c_b

d_a=(c_a1/c_a1.std(dim='time'))
d_b=(c_b1/c_b1.std(dim='time'))

TNI_raw=d_a-d_b
TNI_smooth=TNI_raw.rolling(time=5).mean()

e=(TNI_smooth/TNI_smooth.std(dim='time'))

# -

TNI_smooth.mean(dim='ensemble').plot()

e.mean(dim='ensemble').plot()

TNI_smooth.std(dim='time')

(soi.mean(dim='ensemble')<-2).dropna(dim='time').sst.plot()







# +
# Random old plotting / Data
# -

#Open Model phyto data
# phy=xr.open_dataset('/scratch1/pit071/CAFE60/phy_15m.nc')
# phy=phy.rename({'yt_ocean':'lat','xt_ocean':'lon'})




# +

#tpca_m_regrid = xr.open_dataset('/scratch1/pit071/CAFE60/TPCA_month_regrid.nc')

#phy['time']=np.array(phy.indexes['time'].to_datetimeindex(), dtype='datetime64[M]')
#phy=phy.sel(time=tpca_m_regrid.time)

tpca_m_regrid.sel(time=slice(np.datetime64('2010-03-24'),np.datetime64('2010-04-05'))).mean(dim=['time']).tpca.plot(vmin=0,vmax=0.5),plt.show()
phy.sel(time=slice(np.datetime64('2010-03-01'),np.datetime64('2010-04-01')),ensemble=88).mean(dim=['time']).phy.plot(vmin=0,vmax=0.5),plt.show()
((phy.sel(time=slice(np.datetime64('2010-03-01'),np.datetime64('2010-04-01')),ensemble=88).mean(dim=['time']).phy*893.51)/1000).plot(vmin=0,vmax=0.5),plt.show()


a=tpca_m_regrid.sel(time=slice(np.datetime64('2010-03-24'),np.datetime64('2010-04-05'))).mean(dim=['time']).tpca
b=phy.sel(time=slice(np.datetime64('2010-03-01'),np.datetime64('2010-04-01'))).mean(dim=['time','ensemble']).phy 
b1=phy.sel(time=slice(np.datetime64('2010-03-01'),np.datetime64('2010-04-01')),ensemble=92).mean(dim=['time']).phy 
(a-b).plot(vmin=-0.4,vmax=0.4,cmap='bwr')
plt.show()

(((b*893.51)/1000)).plot()#(vmin=-0.2,vmax=0.2)
plt.show()
