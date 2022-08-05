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

# +
# # %load C60_jupyter_imports.py
# %load_ext autoreload
# %autoreload 2

import xarray as xr
import numpy as np
from dask.distributed import Client
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import xesmf as xe
from scipy.stats import linregress
import os
import requests
import os
import scipy.signal as sps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
# Lets import our functions. There are no tests here. Stored separately so as to not make a mess.
# Make sure in right dir first
# #!pip install --user seawater==3.3.4
import seawater

os.chdir('/g/data/xv83/np1383/src_CAFE60_eqpac_analysis_code/')
from C60_obs_functions import convert_trim_fratios
from xarray_trends import xarray_detrend,xarray_get_trend, deseasonaliser
from dask.distributed import Client,Scheduler
from dask_jobqueue import SLURMCluster
# -



# + [markdown] tags=[]
# ### Plan
#
# Need to compare New Production, CO2 flux and SST at different locations and for different timescale resolutions. 
# Assess which is the best Ensemble for Each of these.
#
#
# What do we need
#
# - Work in mols.
#
#
# - CAFE New Production (Need to make detritus calculation)
# - CO2 flux 
#
# - New Production estimates
# - Chlor A in mg CHL? 
# - Landschutzer CO2 flux estimate (How about DELTA pCO2?)
# - Reynolds OISST product
#
#
# - And then detrend and deseasonalise to get different data product resolutions.
#
# - Plot all ensembles over a Year?
#


# +
#client
# -

# Assumes data was loaded previously in 12 Validation run
use_dask=True
run_chl=False
ensemble=25


if use_dask==True:
    # Set up the remote dask cluster. Can either use this version or a similar version above if building a LocalCluster.
    
    cluster = SLURMCluster(cores=8,processes=2,memory="47GB")
    #cluster = SLURMCluster(cores=8,processes=2,memory="47GB")
    client = Client(cluster)
    cluster.scale(cores=32)
    #cluster.adapt(minimum=2, maximum=16)
    cluster
    #client

cluster


client


# ls /g/data/xv83/dcfp/CAFE60v1/



# +
# Load the data we created

cafe_u=xr.open_dataset('../processed_data/physics_ds/u_physics_feb7.nc').u
cafe_v=xr.open_dataset('../processed_data/physics_ds/v_physics_feb7.nc').v
cafe_wt=xr.open_dataset('../processed_data/physics_ds/wt_physics_feb7.nc').wt
cafe_temp=xr.open_dataset('../processed_data/physics_ds/temp_physics_feb7.nc').temp
cafe_sst=xr.open_dataset('../processed_data/physics_ds/sst_physics_feb7.nc').sst
cafe_age=xr.open_dataset('../processed_data/physics_ds/age_global_physics_feb7.nc').age_global

dic=xr.open_dataset('../processed_data/physics_ds/dic_pac_feb7.nc') #physics_ds
upwelling_cafe=cafe_wt.sel(sw_ocean=-100,method='nearest')

adic_cafe=dic['adic']
dic_cafe=dic['dic']
anth_dic_cafe=xr.open_dataset('../processed_data/physics_ds/anth_DIC_pac_feb7.nc')

cafe_pprod=xr.open_dataset('../processed_data/physics_ds/pprod_gross_2d_bgc_feb7.nc').pprod_gross_2d
cafe_co2=xr.open_dataset('../processed_data/physics_ds/stf10_bgc_feb7.nc').stf10
cafe_co2_natural=xr.open_dataset('../processed_data/physics_ds/stf07_bgc_feb7.nc').stf07
cafe_pCO2=xr.open_dataset('../processed_data/physics_ds/pco2_bgc_feb7.nc').pco2
cafe_paCO2=xr.open_dataset('../processed_data/physics_ds/paco2_bgc_feb7.nc').paco2
cafe_co2flux=xr.open_dataset('../processed_data/physics_ds/stf10_bgc_feb7.nc').stf10
cafe_natco2flux=xr.open_dataset('../processed_data/physics_ds/stf07_bgc_feb7.nc').stf07
#age=xr.open_dataset('../processed_data/physics_ds/pCO2_bgc_feb07.nc').age_global
# -
cafe_v

anth_dic_cafe.chunk('auto').interp(lat=cafe_v.lat_x)

#Need to run these all seperataly and manually because these break (worker killed) if run in succession.
save_dic_transport=True
run_anth=True
if save_dic_transport==True:
    cafe_v1=cafe_v
    cafe_v1=cafe_v1.rename({'lat_x':'lat','lon_x':'lon'})
    cafe_v1=cafe_v1.sel(lat=cafe_v1.lat[:-1],lon=cafe_v1.lon[:-1])

    #cafe_v1['lat']=cafe_v1['lat'][0:-1]
    #cafe_v1['lon']=cafe_v1['lon'][0:-1]

    v1=cafe_v1.chunk('auto')#.mean(dim='time')
    v1['lat']=adic_cafe.lat
    v1['lon']=adic_cafe.lon
    #dic_tx=v1.sel(time=slice('2010','2020')).mean(dim='time')*adic_cafe.sel(time=slice('2010','2020')).mean(dim='time')#.mean(dim='time')
    
    if run_anth==True:
        dic_tx_v=v1*anth_dic_cafe
        print(dic_tx_v)#.chunks)
        dic_tx_v.chunk('auto').to_netcdf('../processed_data/dic_v_anth.nc')
        print('saved')
    else:
        dic_tx_v1=v1*adic_cafe
        print(dic_tx_v1.chunks)
        print(dic_tx_v1)
        dic_tx_v1.chunk('auto').to_netcdf('../processed_data/dic_v.nc')
        print('saved')



#dic_tx_v.to_netcdf('../processed_data/dic_v.nc')
save_dic_transport=False
save_dic_transport=False
if save_dic_transport==True:
    # U CURRENTS
    cafe_u1=cafe_u
    cafe_u1=cafe_u1.rename({'lat_x':'lat','lon_x':'lon'})
    
      cafe_u1=cafe_u1.chunk('auto').interp(lat=cafe_potential_density.lat) # U is X (lon)
        tracer_v_interpolate=tracer.chunk('auto').interp(lon=cafe_potential_density.lon) # V is Y (lat)
        
        
    
    cafe_u1=cafe_u1.sel(lat=cafe_u1.lat[:-1],lon=cafe_u1.lon[:-1])

    #cafe_v1['lat']=cafe_v1['lat'][0:-1]
    #cafe_v1['lon']=cafe_v1['lon'][0:-1]

    u1=cafe_u1.chunk('auto')#.mean(dim='time')
    u1['lat']=adic_cafe.lat
    u1['lon']=adic_cafe.lon

    if run_anth==True:
        dic_tx_u=u1*anth_dic_cafe#.sel(time=slice('2010','2020')).mean(dim='time')
        print(dic_tx_u.chunks)
        print(dic_tx_u)
        dic_tx_u.chunk('auto').to_netcdf('../processed_data/dic_u_anth.nc')
        print('saved')
    #dic_txu=dic_tx_u.sel(time=slice('2010','2020')).mean(dim='time')#*adic_cafe.sel(time=slice('2010','2020')).mean(dim='time')#.mean(dim='time')
    else:
        dic_tx_u1=u1*adic_cafe
        print(dic_tx_u1.chunks)
        print(dic_tx_u1)
        
        dic_tx_u1.chunk('auto').to_netcdf('../processed_data/dic_u.nc')
        print('saved')

#dic_tx_v.to_netcdf('../processed_data/dic_v.nc')
save_dic_transport=True
run_anth=False
if save_dic_transport==True:
    # U CURRENTS
    cafe_w1=cafe_wt
    cafe_w1=cafe_w1.rename({'sw_ocean':'st_ocean'})#'lat_x':'lat','lon_x':'lon'})
    cafe_w1.in['st_ocean']=adic_cafe.st_ocean
    #cafe_w1=cafe_w1.sel(lat=cafe_w1.lat[:-1],lon=cafe_w1.lon[:-1])

    #cafe_v1['lat']=cafe_v1['lat'][0:-1]
    #cafe_v1['lon']=cafe_v1['lon'][0:-1]

    w1=cafe_w1.chunk('auto')#.mean(dim='time')
    w1['lat']=adic_cafe.lat
    w1['lon']=adic_cafe.lon

    if run_anth==True:
        dic_tx_w=w1*anth_dic_cafe.chunk('auto')#.sel(time=slice('2010','2020')).mean(dim='time')
        print(dic_tx_w.chunks)
        dic_tx_w.chunk('auto').to_netcdf('../processed_data/dic_w_anth.nc')
        print('saved')
    #dic_txu=dic_tx_u.sel(time=slice('2010','2020')).mean(dim='time')#*adic_cafe.sel(time=slice('2010','2020')).mean(dim='time')#.mean(dim='time')
    else:
        dic_tx_w1=w1*adic_cafe.chunk('auto')
        print(dic_tx_w1.chunks)
        dic_tx_w1.chunk('auto').to_netcdf('../processed_data/dic_w.nc')
        print('saved')
