import xarray as xr
import numpy as np
from dask.distributed import Client
import matplotlib as mpl
from carbon_math import carbon_flux
import xesmf as xe
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xesmf as xe
from scipy.stats import linregress

#Need some better commenting here. How about updating metadata as well?

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
   

def cut_aco2():
    BGC_monthly = xr.open_zarr('/OSM/CBR/OA_DCFP/data/model_output/CAFE/data_assimilation/d60-zarr/ocean_bgc_month.zarr/',consolidated=True)
    BGC_monthly['xt_ocean']=BGC_monthly['xt_ocean']+360
    print(str(BGC_monthly.stf10.sel(xt_ocean=slice(120,290),yt_ocean=slice(-20,20)).nbytes/1e9) + ' GB')
    co2_flux=BGC_monthly.stf10.sel(xt_ocean=slice(120,290),yt_ocean=slice(-20,20))
    co2_flux=(co2_flux/1000)*86400 #mmol/s to mol/day
    co2_flux=co2_flux*-12 #to grams (And convert to ocean minus air not air minus sea. )
    co2_flux=co2_flux.rename({'xt_ocean':'lon','yt_ocean':'lat'})
    
    co2_flux['time']=np.array(co2_flux.indexes['time'].to_datetimeindex(), dtype='datetime64[M]')
    co2_flux=co2_flux.sel(time=slice(np.datetime64('1982-01-01'),np.datetime64('2020-01-01')))
    
    co2_flux.to_netcdf('/scratch1/pit071/CAFE60/CO2flux_anth/eqpac.nc')
    
def cut_eq_vars():
    BGC_monthly = xr.open_zarr('/OSM/CBR/OA_DCFP/data/model_output/CAFE/data_assimilation/d60-zarr/ocean_bgc_month.zarr/',consolidated=True)
    BGC_monthly['xt_ocean']=BGC_monthly['xt_ocean']+360
    BGC_monthly=BGC_monthly.sel(xt_ocean=slice(120,290),yt_ocean=slice(-20,20))
    BGC_monthly=BGC_monthly.rename({'xt_ocean':'lon','yt_ocean':'lat'})
    selected=BGC_monthly[['stf10','pprod_gross_2d','export_prod']]#,'phy']]
    
    selected['stf10']=((selected['stf10']/1000)*86400)*-12 #mmol/s to mol/day
    selected['pprod_gross_2d']=(selected['pprod_gross_2d']*6.625*12*86400)/1000
    selected['export_prod']=(selected['export_prod']*6.625*12*86400)/1000
    
    selected['time']=np.array(selected.indexes['time'].to_datetimeindex(), dtype='datetime64[M]')
    selected=selected.sel(time=slice(np.datetime64('1982-01-01'),np.datetime64('2020-01-01')))
    
    print(str(selected.nbytes/1e9) + ' GB')
    selected.to_netcdf('/scratch1/pit071/CAFE60/eqpac_BGC.nc')
    
def proc_landschutzer():
    #Load and process landschutzer data
    landschutzer_CO2=xr.open_dataset('/scratch1/pit071/carbon_data_ch2/spco2_MPI-SOM_FFN_v2020.nc')
    landschutzer_CO2= landschutzer_CO2.assign_coords(lon=(landschutzer_CO2.lon % 360)).roll(lon=(landschutzer_CO2.dims['lon']),roll_coords=False).sortby('lon')		#EPIC 1 line fix for the dateline problem.
    landschutzer_CO2=landschutzer_CO2.sel(lon=slice(120,290),lat=slice(-20,20)).fgco2_smoothed/365 #From per to per day
    landschutzer_CO2=landschutzer_CO2*12 #to grams
    landschutzer_CO2['time']=landschutzer_CO2['time'].astype('datetime64[M]')

    #Regrid
    regridder = xe.Regridder(landschutzer_CO2, CAFE_CO2_flux_anth.stf10, 'bilinear',reuse_weights=True)
    landschutzer_CO2=regridder(landschutzer_CO2)
    landschutzer_CO2.to_netcdf('/scratch1/pit071/carbon_data_ch2/landshutzer_regrid.nc')
    
def proc_landschutzer_old():
    #Load and process landschutzer data
    landschutzer_CO2=xr.open_dataset('/OSM/CBR/OA_DCFP/work/mat236/obs/spco2_clim_1985-2015_MPI_SOM-FFN_v2016.nc')
    landschutzer_CO2= landschutzer_CO2.assign_coords(lon=(landschutzer_CO2.lon % 360)).roll(lon=(landschutzer_CO2.dims['lon']),roll_coords=False).sortby('lon')		#EPIC 1 line fix for the dateline problem.
    landschutzer_CO2=landschutzer_CO2.sel(lon=slice(120,290),lat=slice(-20,20)).fgco2_clim/365 #From per to per day
    landschutzer_CO2=landschutzer_CO2*12 #to grams
    #landschutzer_CO2['time']=landschutzer_CO2['time'].astype('datetime64[M]')

    #Regrid
    regridder = xe.Regridder(landschutzer_CO2, CAFE_CO2_flux_anth.stf10, 'bilinear',reuse_weights=True)
    landschutzer_CO2=regridder(landschutzer_CO2)
    landschutzer_CO2.to_netcdf('/scratch1/pit071/carbon_data_ch2/landshutzer_old_regrid.nc')


def cut_sst(): #And calulate Nino3-4 index.
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

def cut_regrid_sst():
    sst_cafe=xr.open_dataset('/scratch1/pit071/CAFE60/sst.nc').sst
    sst_obs=xr.open_dataset('/scratch1/pit071/carbon_data_ch2/sst.mnmean.nc')
    sst_obs=sst_obs.sst.sel(lat=slice(20,-20),lon=slice(120,290),time=slice('1982-01-01','2019-12-01'))
    sst_obs=sst_obs.reindex(lat=sst_obs.lat[::-1])
    regridder = xe.Regridder(sst_obs, sst_cafe, 'bilinear',reuse_weights=True)
    sst_obs_regrid=regridder(sst_obs)
    sst_obs_regrid.to_netcdf('/scratch1/pit071/CAFE60/sst.mnmean.regrid.nc')