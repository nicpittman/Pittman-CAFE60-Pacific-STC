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
# +
# Similar to 14 except removed all the processing code.


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
ensemble=25


# +

if use_dask==True:
    # Set up the remote dask cluster. Can either use this version or a similar version above if building a LocalCluster.
    
    cluster = SLURMCluster(cores=2,processes=1,memory="16GB")
    #cluster = SLURMCluster(cores=8,processes=2,memory="47GB")
    client = Client(cluster)
    cluster.scale(cores=16)
    #cluster.adapt(minimum=2, maximum=16)
    cluster
    #client
# -

cluster


client


# LOAD CAFE
bgc=xr.open_zarr('/g/data/xv83/dcfp/CAFE60v1/ocean_bgc_month.zarr.zip')
bgc=bgc.rename({'xt_ocean':'lon','yt_ocean':'lat'})
bgc['lon']=bgc['lon']+360
eqpac=bgc.sel(lon=slice(120,290),lat=slice(-40,40))
eqpac['time']=eqpac.time.astype('datetime64[M]')
eqpac

# ls /g/data/xv83/dcfp/CAFE60v1

# +
# Load Physics
phys=xr.open_zarr('/g/data/xv83/dcfp/CAFE60v1/ocean_month.zarr.zip')
phys=phys.rename({'xu_ocean':'lon_x','yu_ocean':'lat_x'})
phys=phys.rename({'xt_ocean':'lon','yt_ocean':'lat'})

phys['lon']=phys['lon']+360
phys['lon_x']=phys['lon_x']+360

phys_eqpac=phys.sel(lon=slice(120,290),lat=slice(-40,40),lon_x=slice(120,290),lat_x=slice(-40,40),)

phys_eqpac['time']=phys_eqpac.time.astype('datetime64[M]')
phys_eqpac
# +
# Load OBS

# Landschutzer
land_co2=xr.open_dataset('../processed_data/obs/landshutzer_global_regrid.nc')/365 #g/m2/day
land_co2=(land_co2.fgco2_smoothed.sel(lon=slice(120,290),lat=slice(-40,40))/12)*1000#/1000 #mmol/m2/day to match cafe. (ingassing is positive, outgassing neg)
#land_dpco2=(land_co2.pco2.sel(lon=slice(120,290),lat=slice(-40,40))


# Reynolds SST
rey_sst=xr.open_dataset('../processed_data/obs/sst.mnmean.regrid.global.nc')
rey_sst=rey_sst.sst.sel(lon=slice(120,290),lat=slice(-40,40))
rey_sst=xr.open_dataset('../processed_data/obs/sst.mnmean.regrid.global.nc').sel(lon=slice(120,290),lat=slice(-40,40))
#(obs_current.U_320.median(dim='time').sel(lat=0,method='nearest')).interpolate_na(dim='depth').plot.contourf(cmap='bwr')
# -
# TOA Moorings
run_obs_current=False
if run_obs_current==True:
    obs_current=xr.open_mfdataset('../external_data/mooring_u_current/*.cdf')
    obs_current=obs_current.where(obs_current<=1e20)/100
    obs_current['time']=obs_current.time.astype('datetime64[M]')
    obs_current.to_netcdf('../processed_data/tao_adcp.nc')
obs_current=xr.open_dataset('../processed_data/tao_adcp.nc')





# +
# Load the data we created

cafe_u=xr.open_dataset('../processed_data/physics_ds/u_physics_feb7.nc').u
cafe_v=xr.open_dataset('../processed_data/physics_ds/v_physics_feb7.nc').v
cafe_wt=xr.open_dataset('../processed_data/physics_ds/wt_physics_feb7.nc').wt
cafe_temp=xr.open_dataset('../processed_data/physics_ds/temp_physics_feb7.nc').temp
cafe_sst=xr.open_dataset('../processed_data/physics_ds/sst_physics_feb7.nc').sst
cafe_age=xr.open_dataset('../processed_data/physics_ds/age_global_physics_feb7.nc').age_global
#cafe_salt=xr.open_dataset('../processed_data/physics_ds/salt_physics_feb7.nc').salt
cafe_density=xr.open_dataset('../processed_data/physics_ds/density_physics_feb7.nc').__xarray_dataarray_variable__
cafe_potential_density=xr.open_dataset('../processed_data/physics_ds/potential_density_physics_feb7.nc').__xarray_dataarray_variable__
dic=xr.open_dataset('../processed_data/dic_pac_feb7.nc') #physics_ds
upwelling_cafe=cafe_wt.sel(sw_ocean=-100,method='nearest')

cafe_u_tx=xr.open_dataset('../processed_data/physics_ds/tx_trans_physics_feb7.nc').tx_trans
cafe_y_tx=xr.open_dataset('../processed_data/physics_ds/ty_trans_physics_feb7.nc').ty_trans
cafe_u_tx_gm=xr.open_dataset('../processed_data/physics_ds/tx_trans_gm_physics_feb7.nc').tx_trans_gm # Including eddies or subscale processes?
cafe_y_tx_gm=xr.open_dataset('../processed_data/physics_ds/ty_trans_gm_physics_feb7.nc').ty_trans_gm


adic_cafe=dic['adic']
dic_cafe=dic['dic']
anth_dic_cafe=xr.open_dataset('../processed_data/physics_ds/anth_DIC_pac_feb7.nc').__xarray_dataarray_variable__


cafe_pprod=xr.open_dataset('../processed_data/physics_ds/pprod_gross_2d_bgc_feb7.nc').pprod_gross_2d
cafe_co2=xr.open_dataset('../processed_data/physics_ds/stf10_bgc_feb7.nc').stf10
cafe_co2_natural=xr.open_dataset('../processed_data/physics_ds/stf07_bgc_feb7.nc').stf07
cafe_co2_anth=cafe_co2-cafe_co2_natural

cafe_pCO2=xr.open_dataset('../processed_data/physics_ds/pco2_bgc_feb7.nc').pco2
cafe_paCO2=xr.open_dataset('../processed_data/physics_ds/paco2_bgc_feb7.nc').paco2
#cafe_co2flux=xr.open_dataset('../processed_data/physics_ds/stf10_bgc_feb7.nc').stf10
#cafe_natco2flux=xr.open_dataset('../processed_data/physics_ds/stf07_bgc_feb7.nc').stf07
#age=xr.open_dataset('../processed_data/physics_ds/pCO2_bgc_feb07.nc').age_global


dic_tx_v=xr.open_dataset('../processed_data/dic_v.nc').__xarray_dataarray_variable__
dic_tx_u=xr.open_dataset('../processed_data/dic_u.nc').__xarray_dataarray_variable__
#dic_tx_w=xr.open_dataset('../processed_data/dic_w.nc').__xarray_dataarray_variable__

anth_dic_tx_v=xr.open_dataset('../processed_data/dic_v_anth.nc').__xarray_dataarray_variable__
anth_dic_tx_u=xr.open_dataset('../processed_data/dic_u_anth.nc').__xarray_dataarray_variable__
#anth_dic_tx_w=xr.open_dataset('../processed_data/dic_w_anth.nc').__xarray_dataarray_variable__


# TREND FILES
cafe_co2_natural_trend=xr.open_dataset('../processed_data/var_ensembles/cafe_co2_natural_trend_ensemble.nc')['1998-01 to 2015-01']
cafe_co2_trend=xr.open_dataset('../processed_data/var_ensembles/cafe_co2_trend_ensemble.nc')['1998-01 to 2015-01']
cafe_sst_trend=xr.open_dataset('../processed_data/var_ensembles/cafe_sst_trend_ensemble.nc')['1998-01 to 2015-01']
cafe_u_trend=xr.open_dataset('../processed_data/var_ensembles/cafe_u_mean_depth_trend_ensemble.nc')['1998-01 to 2015-01']
cafe_v_trend=xr.open_dataset('../processed_data/var_ensembles/cafe_v_mean_depth_trend_ensemble.nc')['1998-01 to 2015-01']
land_co2_trend=xr.open_dataset('../processed_data/var_ensembles/land_co2_trend_ensemble.nc')['1998-01 to 2015-01']
rey_sst_trend=xr.open_dataset('../processed_data/var_ensembles/rey_sst.sst_trend_ensemble.nc')['1998-01 to 2015-01']
tao_u_trend=xr.open_dataset('../processed_data/var_ensembles/obs_current.U_320_trend_ensemble.nc')['1998-01 to 2015-01']
tao_v_trend=xr.open_dataset('../processed_data/var_ensembles/obs_current.V_321_trend_ensemble.nc')['1998-01 to 2015-01']
dic_mean_depth_trend=xr.open_dataset('../processed_data/var_ensembles/dic_mean_depth_trend_ensemble.nc')['1998-01 to 2015-01']
adic_mean_depth_trend=xr.open_dataset('../processed_data/var_ensembles/adic_mean_depth_trend_ensemble.nc')['1998-01 to 2015-01']
upwelling_cafe_trend=xr.open_dataset('../processed_data/var_ensembles/cafe_pprod_trend_ensemble.nc')['1998-01 to 2015-01']
cafe_pprod_trend=xr.open_dataset('../processed_data/var_ensembles/upwelling_cafe_trend_ensemble.nc')['1998-01 to 2015-01']
npp_cafe_trend=xr.open_dataset('../processed_data/var_ensembles/npp_trend_ensemble.nc')['1998-01 to 2015-01']


co2_rodenbeck=(xr.open_dataset('../processed_data/obs/rodenbeck_global_regrid.nc').sel(lon=slice(120,290),lat=slice(-40,40)).co2flux_ocean/12)*1000 #to mmolC


# +
#Calculate ENSO

#Process EP, CP and Nino events.
elnino=pd.read_csv('../processed_data/indexes/el_nino_events_ch2.csv')
lanina=pd.read_csv('../processed_data/indexes/la_nina_events_ch2.csv')
ep_nino=pd.read_csv('../processed_data/indexes/ep_events_ch2.csv')
cp_nino=pd.read_csv('../processed_data/indexes/cp_events_ch2.csv')


nina=pd.DataFrame()
#nino=pd.DataFrame() @Wrapped in ep and cp... but maybe good to have own months?
ep=pd.DataFrame()
cp=pd.DataFrame()
all_dates=eqpac.sel(time=slice('1980','2020')).time#cafe_sst_mean.time

#Set to after 2000.
all_dates=all_dates.where(all_dates.time>=np.datetime64('2000-01-01')).dropna(dim='time')
all_dates=all_dates.where(all_dates.time<np.datetime64('2020-01-01')).dropna(dim='time')

def drop_df_values(df1,df2):
    df=df1[~df1.isin(df2)].dropna(how = 'all')
    return df

#for i in lanina.iterrows(): nina=nina.append(info[slice(i[1].start,i[1].end)])
for i in lanina.iterrows(): nina=nina.append(list(all_dates.sel(time=slice(i[1].start,i[1].end)).time.values))
for i in ep_nino.iterrows(): ep=ep.append(list(all_dates.sel(time=slice(i[1].start,i[1].end)).time.values))
for i in cp_nino.iterrows(): cp=cp.append(list(all_dates.sel(time=slice(i[1].start,i[1].end)).time.values))

#all_dates=chl.time
all_dates2=pd.DataFrame(all_dates.values)#[36:] #2000 - 2020
neutral=drop_df_values(drop_df_values(drop_df_values(all_dates2[0],cp[0]),ep[0]),nina[0])
#ep,cp,nino,neutral,info

cp_events=cp[0].values
ep_events=ep[0].values
nina_events=nina[0].values
neutral_events=neutral.values
# -



# cafe_u=xr.open_dataset('../processed_data/physics_ds/u_physics_feb7.nc').u
# cafe_v=xr.open_dataset('../processed_data/physics_ds/v_physics_feb7.nc').v
# cafe_wt=xr.open_dataset('../processed_data/physics_ds/wt_physics_feb7.nc').wt
# cafe_temp=xr.open_dataset('../processed_data/physics_ds/temp_physics_feb7.nc').temp
# cafe_sst=xr.open_dataset('../processed_data/physics_ds/sst_physics_feb7.nc').sst
# cafe_age=xr.open_dataset('../processed_data/physics_ds/age_global_physics_feb7.nc').age_global
# cafe_salt=xr.open_dataset('../processed_data/physics_ds/salt_physics_feb7.nc').salt
# cafe_density=xr.open_dataset('../processed_data/physics_ds/density_physics_feb7.nc').__xarray_dataarray_variable__
# cafe_potential_density=xr.open_dataset('../processed_data/physics_ds/potential_density_physics_feb7.nc').__xarray_dataarray_variable__
# dic=xr.open_dataset('../processed_data/dic_pac_feb7.nc') #physics_ds
# upwelling_cafe=cafe_wt.sel(sw_ocean=-100,method='nearest')
#
# cafe_u_tx=xr.open_dataset('../processed_data/physics_ds/tx_trans_physics_feb7.nc').tx_trans
# cafe_y_tx=xr.open_dataset('../processed_data/physics_ds/ty_trans_physics_feb7.nc').ty_trans
# cafe_u_tx_gm=xr.open_dataset('../processed_data/physics_ds/tx_trans_gm_physics_feb7.nc').tx_trans_gm # Including eddies or subscale processes?
# cafe_y_tx_gm=xr.open_dataset('../processed_data/physics_ds/ty_trans_gm_physics_feb7.nc').ty_trans_gm
#


pwd

# +
# I cut out ensemble 25 previously, path is '/g/data4/xv83/np1383/processed_data/physics_ds/ .... tx_trans_physics_feb7.nc

dic=xr.open_dataset('../processed_data/dic_pac_feb7.nc') #physics_ds


cafe_x_tx=xr.open_dataset('../processed_data/physics_ds/tx_trans_physics_feb7.nc').tx_trans
cafe_y_tx=xr.open_dataset('../processed_data/physics_ds/ty_trans_physics_feb7.nc').ty_trans
cafe_x_tx_gm=xr.open_dataset('../processed_data/physics_ds/tx_trans_gm_physics_feb7.nc').tx_trans_gm # Including eddies or subscale processes?
cafe_y_tx_gm=xr.open_dataset('../processed_data/physics_ds/ty_trans_gm_physics_feb7.nc').ty_trans_gm
# y is v and x is u directions?


adic_cafe=dic['adic']
dic_cafe=dic['dic']
anth_dic_cafe=xr.open_dataset('../processed_data/physics_ds/anth_DIC_pac_feb7.nc').__xarray_dataarray_variable__


# +
# Area grid  for Depth??
# #!ls /g/data/xv83/rxm599
#/area.nc
# 

#docn=phys
#mdepth=docn.st_ocean.copy()
#dbot=np.copy(docn.sw_ocean)
#dthick=dbot*0
#dthick[1:50]=dbot[1:50]-dbot[0:49]

#dthick[0]=dbot[0]
#print(dthick,dbot)
#mdepth=mdepth*0+dthick
#mdepth 
# -
adic_cafe
adic_cafe_x=adic_cafe.rename({'lat':'lat_x'}).interp(lat_x=cafe_y_tx.lat_x)
adic_cafe_x
cafe_x_tx
cafe_y_tx
cafe_u_tx=adic_cafe.interp(lon=cafe_x_tx.lon_x)#cafe_x_tx
cafe_x_tx
cafe_co2


# +
# Calculate U_TX * anthDIC
u1=(cafe_u_tx.chunk('auto')).rolling(lon_x=2).mean()
anth_dic_cafe_u=(adic_cafe.chunk('auto')).rolling(lat=2).mean()
u1=u1.rename({'lon_x':'lon'})
anth_dic_cafe_u['lon']=u1.lon[:-1]
adic_utx=anth_dic_cafe_u*u1

# Calculate V (or Y)_TX * anthDIC
v1=(cafe_y_tx.chunk('auto')).rolling(lat_x=2).mean()
anth_dic_cafe_v=(adic_cafe.chunk('auto')).rolling(lon=2).mean()
v1=v1.rename({'lat_x':'lat'})
anth_dic_cafe_v['lat']=v1.lat[:-1]
adic_vtx=anth_dic_cafe_v*v1

# Calculate U_TX * DIC
u1=(cafe_u_tx.chunk('auto')).rolling(lon_x=2).mean()
dic_cafe_u=(dic_cafe.chunk('auto')).rolling(lat=2).mean()
u1=u1.rename({'lon_x':'lon'})
dic_cafe_u['lon']=u1.lon[:-1]
dic_utx=dic_cafe_u*u1

# Calculate V (or Y)_TX * DIC
v1=(cafe_y_tx.chunk('auto')).rolling(lat_x=2).mean()
dic_cafe_v=(adic_cafe.chunk('auto')).rolling(lon=2).mean()
v1=v1.rename({'lat_x':'lat'})
dic_cafe_v['lat']=v1.lat[:-1]
dic_vtx=dic_cafe_v*v1

# Units should be  mmolC / m^3 / (Sv) 10^9 kg/s = 1,000,000 m3/sec

adic_utx.mean(dim='time').sel(lat=-3,method='nearest').plot()
plt.show()

adic_vtx.mean(dim='time').sel(lat=-3,method='nearest').plot(vmin=-500,vmax=500,cmap='bwr')
plt.show()
# -



adic_vtx.mean(dim='time').sel(lat=-3,method='nearest').plot(vmin=-500,vmax=500,cmap='bwr')

anth_dic_cafe_v

v1   # mmolC / m^3 / (Sv) 10^9 kg/s = 1,000,000 m3/sec

v1.mean(dim='time').sel(lat=0,method='nearest').plot()

u1.mean(dim='time').sel(lat=0,method='nearest').plot()



test1=(cafe_u_tx.chunk('auto')).mean(dim='time').sel(lat=-3,method='nearest')#.plot()
test1=(adic_cafe.chunk('auto')).mean(dim='time').sel(lat=-3,method='nearest')#.plot()
#test1=(cafe_u_tx.chunk('auto')).mean(dim='time').sel(lat=-3,method='nearest')#.plot()

test

test.plot()









# +
# Calculate M2 grid 



def make_earth_grid_m2():
    boxlo,boxla=np.array(np.meshgrid(np.arange(0.5,359.5,1),np.arange(-89.5,89.5,1)))
    actual_grid=np.cos(np.radians(abs(boxla)))*(111.1*111.1*1000*1000)
    grid_nc=xr.DataArray(actual_grid,coords={'lat':boxla[:,1],'lon':boxlo[1,:]},dims=['lat','lon'])
    lat_size=110567 #in m
    grid_nc['area']=grid_nc#*lat_size
    grid_nc1=grid_nc['area']
    grid_nc1.name='m2'
    #grid_nc.to_netcdf('processed/earth_m2.nc',engine='h5netcdf',mode='w')
    return grid_nc1


#grid.name='test'
#grid
model_dummy=phys_eqpac.temp
depths=model_dummy.sel(st_ocean=slice(0,500)).st_ocean.values
grid=make_earth_grid_m2().sel(lat=slice(-40,40),lon=slice(120,290)).to_dataset()
#grid.assign_coords(depth=depths.values)#['depths']=depths.values

#the m2 is actually wrong so regrid and adjust to cafe grid
rg=xe.Regridder(grid,model_dummy,"bilinear")# ds,dsout
rg_m=rg(grid)
grid_rg=(rg_m*(len(grid.lat)/len(model_dummy.lat))) # Normalise the m2 cells to their new sizes. 
grid_rg['lon_m']=[111100]*len(grid_rg.lon)#m, 111.11km

rglats=np.diff(grid_rg.lat)
rglats=np.append(rglats,rglats[-1]) #so they match size
                 
grid_rg['lat_m']=([111100]*len(grid_rg.lat))*rglats #m, 111.11km

grid_rg['depth']=-depths##.m2#.to_dataset()
l=0
depth_diffs=[]
# Calcualate the m between depths so we can integrate
for i,x in enumerate(grid_rg.depth):
    d=x.values-l
    l=x.values
    #print(d)#x.values)
    depth_diffs.append(d)
grid_rg['depth_diff']=depth_diffs
grid_rg['depth_m3']=grid_rg.m2*grid_rg.depth_diff
grid_rg['depth_lat_m2']=grid_rg.lat_m*grid_rg.depth_diff#np.sqrt(grid_rg['depth_m3']) # This should be precise enough on the equator right? Imagine further off boxes arent square.
grid_rg['depth_lon_m2']=grid_rg.lon_m*grid_rg.depth_diff#np.sqrt(grid_rg['depth_m3']) # This should be precise enough on the equator right? Imagine further off boxes arent square.

#grid_rg['name']='gridsize'
grid_rg

#np.diff(grid.depth.values,axis=0)#()#help(dep)  #.map()#grid.assign_coords(depths=depths)
#gridm2=grid.to_dataset()
#depths  
# -
grid_rg


# ## Add a few custom functions ive been making.
# wrap functions to wrap functions that wrap functions
# who wraps the wrapper for the wrapper? 

# + tags=[]



# + tags=[]

def plot_pacific(dat,levels=None,cmap='bwr'):

    fig = plt.figure(figsize=(12,7))

    # this declares a recentered projection for Pacific areas
    proj = ccrs.PlateCarree(central_longitude=180)
    proj._threshold /= 20.  # to make greatcircle smooth

    ax = plt.axes(projection=proj)
    # set appropriate extents: (lon_min, lon_max, lat_min, lat_max)
    ax.set_extent([120, 290, -20, 20], crs=ccrs.PlateCarree())

    geodetic = ccrs.Geodetic()
    plate_carree = ccrs.PlateCarree(central_longitude=180)

    lonm,latm=np.meshgrid(dat.lon,dat.lat)
    if levels is None:
        g=ax.contourf(dat.lon,dat.lat,dat, transform=ccrs.PlateCarree(),cmap=cmap)
    else:
        g=ax.contourf(dat.lon,dat.lat,dat, transform=ccrs.PlateCarree(),cmap=cmap,levels=levels)
    plt.colorbar(g,ax=ax,location='bottom',orientation='horizontal')
    # plot greatcircle arc

    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100, edgecolor='k')
    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.coastlines()

    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3)

    plt.show()
    
    
# #Seasonality?
#plot_pacific(sst_rey.groupby('time.month').mean().std(dim=['month']))

#stat=xarray_get_trend(rey_sst)
#plot_pacific(stat.sel(parameter=0)*365)

#stat=xarray_get_trend(sst_cafe.sel(ensemble=25))
#plot_pacific(stat.sel(parameter=0)*365)

# + tags=[]
def plot_pacific_subplot(dat,sb,title,levels=None,units=None,cmap='bwr',extend='neither',shrink=0.85,small_plot=False,remap=False):
   
    # this declares a recentered projection for Pacific areas
    if remap==False:
        proj = ccrs.PlateCarree(central_longitude=180)
        proj1=ccrs.PlateCarree()
        
    #elif remap==True:
    #    proj = ccrs.Miller(central_longitude=180)
    #    proj1=ccrs.Miller()


    if isinstance(sb,int):
        ax=plt.subplot(sb,projection=proj)
    else:
        ax=plt.subplot(sb[0],sb[1],sb[2],projection=proj)
    # set appropriate extents: (lon_min, lon_max, lat_min, lat_max)
    if small_plot==True:
        ax.set_extent([120, 290, -15, 15], crs=proj1)
    elif small_plot==False:    
        ax.set_extent([120, 290, -40, 40], crs=proj1)

    geodetic = ccrs.Geodetic()
    plate_carree = ccrs.PlateCarree(central_longitude=180)

    lonm,latm=np.meshgrid(rey_sst.lon,rey_sst.lat)
    if isinstance(levels,type(None)):
         g=ax.contourf(dat.lon,dat.lat,dat, transform=proj1,cmap=cmap, extend=extend)
    else:
         g=ax.contourf(dat.lon,dat.lat,dat, transform=proj1,cmap=cmap,levels=levels, extend=extend)
   #,levels=levels)#vmin=vmin,vmax=vmax)
    #ax.clim(vmin,vmax)
    cb=plt.colorbar(g,ax=ax,location='bottom',orientation='horizontal',shrink=shrink)#,vmin=vmin,vmax=vmax)
    if units is not None:
        cb.set_label(units)#, rotation=270)
        
    # plot greatcircle arc
    #ax.set_clim(vmin,vmax)
    #plt.clim(vmin,vmax)
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100, edgecolor='k')
    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.coastlines()
    ax.set_title(title)
    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3)
    ax.set_aspect('auto')


# + tags=[]



# + tags=[]



# + [markdown] tags=[]
# # NEW FIGURES HERE
#
# Mean Left Tr Right
#
#
# - Rey SST:   
#     - rey_sst
# - CAFE SST
#     - sst_cafe
# - Mooring u:           
#     - obs_current.U_320
# - Cafe u
# - Mooring V:           
#     - obs_current.V_321
# - Cafe v
# - Upwelling at 50m 
#


# + tags=[]
# Unsure how to reduce the whitespace between figures? 


# + tags=[]



# + tags=[]
run_f1=True
run_f2=True


# + [markdown] tags=[]
# # Figure 1


# + tags=[]
run_f1=True
if run_f1==True:
    sday='2000-01-01'
    eday='2020-01-01'

    # A3 is (11.69,16.53)
    #fig = plt.figure(figsize=((8.27)*1.5,11.69*1.5)) #Inches Portrait
    shrink=0.8
    fig = plt.figure(figsize=((8.27*2),(11.69*2))) #Inches Portrait

    plot_pacific_subplot(cafe_sst.sel(time=slice(sday,eday)).mean(dim='time'),sb=721,title='Mean SST Cafe',levels=np.arange(14,32,1),units='Degrees C',cmap='viridis',shrink=shrink)
    plot_pacific_subplot(cafe_sst_trend.mean(dim='timeperiod').sel(parameter=0),sb=722,title='Mean SST Cafe Trend',levels=np.arange(-0.1,0.11,0.01),units='Degrees C / Yr',shrink=shrink)

    plot_pacific_subplot(rey_sst.sst.sel(time=slice(sday,eday)).mean(dim='time'),sb=723,title='Mean SST Reynolds',levels=np.arange(14,32,1),cmap='viridis',units='Degrees C',shrink=shrink)
    plot_pacific_subplot(rey_sst_trend.mean(dim='timeperiod').sel(parameter=0),sb=724,title='Mean SST Reynolds Trend',levels=np.arange(-0.1,0.11,0.01),units='Degrees C / Yr',shrink=shrink)

    plot_pacific_subplot(cafe_u.chunk({'st_ocean':-1}).mean('st_ocean').rename({'lon_x':'lon','lat_x':'lat'}).sel(time=slice(sday,eday)).mean(dim='time'),sb=725,title='U Cafe',units='m/s',levels=np.arange(-1,1.1,0.1),shrink=shrink)
    plot_pacific_subplot(cafe_u_trend.mean(dim='timeperiod').sel(parameter=0).rename({'lon_x':'lon','lat_x':'lat'}),sb=726,title='U Cafe Trend',units='m/s/yr',levels=np.arange(-0.02,0.021,0.001),shrink=shrink)

    plot_pacific_subplot(obs_current.U_320.interpolate_na(dim='lat').interpolate_na(dim='lon').mean(dim=['depth','time']),sb=727,levels=np.arange(-1,1.1,0.1),cmap='bwr',title='U TAO Obs',shrink=shrink,small_plot=False)
    plot_pacific_subplot(obs_current.V_321.interpolate_na(dim='lat').interpolate_na(dim='lon').mean(dim=['depth','time']),cmap='bwr',levels=np.arange(-0.1,0.11,0.01),title='V TAO Obs',sb=[7,2,8],shrink=shrink,small_plot=False)


    plot_pacific_subplot(cafe_v.chunk({'st_ocean':-1}).rename({'lon_x':'lon','lat_x':'lat'}).mean('st_ocean').sel(time=slice(sday,eday)).mean(dim='time'),sb=729,title='V Cafe',units='m/s',levels=np.arange(-0.1,0.11,0.01),extend='both',shrink=shrink)
    plot_pacific_subplot(cafe_v_trend.mean(dim='timeperiod').sel(parameter=0).rename({'lon_x':'lon','lat_x':'lat'}),sb=[7,2,10],title='V Cafe Trend',units='m/s/yr',levels=np.arange(-0.002,0.0021,0.0001),shrink=shrink)


    plot_pacific_subplot(upwelling_cafe.sel(time=slice(sday,eday)).mean(dim='time')*60*60*24,sb=[7,2,11],title='Upwelling at 100m  Cafe',units='m/day',levels=np.arange(-1,1.1,0.1),extend='max',shrink=shrink)
    plot_pacific_subplot(upwelling_cafe_trend.mean(dim='timeperiod').sel(parameter=0)*60*60*24,sb=[7,2,12],title='Upwelling Cafe Trend',units='m/day/yr',levels=np.arange(-0.15,0.16,0.01),extend='max',shrink=shrink)
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0)



# + tags=[]



# + tags=[]



# + tags=[]
#(land_co2_trend.mean(dim='timeperiod').sel(parameter=0)*1000).plot()


# + tags=[]
run_f2=True
shrink=0.8


# + tags=[]
cluster


# + [markdown] tags=[]
# # Figure 3 ? 2 for now.


# + tags=[]
if run_f2==True:
    sday='2000-01-01'
    eday='2020-01-01'
    fig = plt.figure(figsize=((8.27*2),11.69*2)) #Inches Portrait

    plot_pacific_subplot(cafe_co2.sel(time=slice(sday,eday)).mean(dim='time')*60*60*-24,sb=721,title='CAFE CO2',levels=np.arange(-20,21,1),units='mmol/m3/day',shrink=shrink,extend='max')
    plot_pacific_subplot(cafe_co2_trend.mean(dim='timeperiod').sel(parameter=0)*60*60*-24,sb=722,title='CAFE CO2 Trend',levels=np.arange(-0.75,0.8,0.05),units='mmol/m3/day/yr',shrink=shrink,extend='max')

    plot_pacific_subplot((land_co2).sel(time=slice(sday,eday)).mean(dim='time'),sb=723,title='Land CO2',levels=np.arange(-20,21,1),units='mmol/m3/year',shrink=shrink)
    plot_pacific_subplot(land_co2_trend.mean(dim='timeperiod').sel(parameter=0)*1000,sb=724,title='Land CO2 Trend',levels=np.arange(-0.75,0.8,0.05),units='mmol/m3/yr',shrink=shrink)

    plot_pacific_subplot(cafe_co2_natural.sel(time=slice(sday,eday)).mean(dim='time')*60*60*-24,sb=725,title='Natural CO2',levels=np.arange(-20,21,1),units='mmol/m3',extend='max',shrink=shrink)
    plot_pacific_subplot(cafe_co2_natural_trend.mean(dim='timeperiod').sel(parameter=0)*60*60*-24,sb=726,title='Natural CO2',levels=np.arange(-0.75,0.8,0.05),units='mmol/m3/yr',extend='max',shrink=shrink)

    plot_pacific_subplot(adic_cafe.sel(st_ocean=slice(0,-100)).mean(dim='st_ocean').sel(time=slice(sday,eday)).mean(dim='time'),sb=727,title='aDIC',levels=np.arange(1800,2240,20),units='mmol/m3',cmap='viridis',shrink=shrink)
    plot_pacific_subplot(adic_mean_depth_trend.mean(dim='timeperiod').sel(parameter=0),sb=728,title='aDIC',levels=np.arange(-9,9.25,0.25),units='mmol/m3/yr',shrink=shrink)#,levels=np.arange(-0.025,0.025,0.0025))

    plot_pacific_subplot(dic_cafe.sel(st_ocean=slice(0,-100)).mean(dim='st_ocean').sel(time=slice(sday,eday)).mean(dim='time'),sb=729,title='DIC',levels=np.arange(1800,2240,20),units='mmol/m3',cmap='viridis',shrink=shrink)
    plot_pacific_subplot(dic_mean_depth_trend.mean(dim='timeperiod').sel(parameter=0),sb=[7,2,10],title='DIC',levels=np.arange(-9,9.25,0.25),units='mmol/m3/yr',shrink=shrink)


    plot_pacific_subplot(cafe_pprod.sel(time=slice(sday,eday)).mean(dim='time')*60*60*24*6.625 ,sb=[7,2,11],title='Cafe Primary Prod',units='mmolC/m3/da',cmap='viridis',levels=np.arange(0,70,5),shrink=shrink)
    plot_pacific_subplot(cafe_pprod_trend.mean(dim='timeperiod').sel(parameter=0)*60*60*24*6.625 ,sb=[7,2,12],title='Cafe Primary Prod Trend',units='mmolC/m3/day/yr',levels=np.arange(-0.5,0.55,0.05),shrink=shrink)

    plot_pacific_subplot(npp.sel(time=slice(sday,eday)).mean(dim='time')/12,sb=[7,2,13],title='CAFE Satellite NPP',units='mmolC/m3/day',cmap='viridis',levels=np.arange(0,70,5),extend='max',shrink=shrink)#,remap=True)
    plot_pacific_subplot(npp_cafe_trend.mean(dim='timeperiod').sel(parameter=0)/12,sb=[7,2,14],title='CAFE Satellite NPP Trends',units='mmolC/m3/day/yr',levels=np.arange(-0.5,0.55,0.05),extend='both',shrink=shrink)#,remap=True)
    #plot_pacific_subplot(xarray_get_trend(npp.sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=[7,2,14],title='CAFE Satellite NPP Trends',units='mmol/m3/yr')

    plt.tight_layout()
    #plot_pacific_subplot(cafe_v.chunk({'st_ocean':-1}).rename({'lon_x':'lon','lat_x':'lat'}).mean('st_ocean').sel(time=slice(sday,eday)).mean(dim='time'),sb=527,title='V Cafe')
    #plot_pacific_subplot(xarray_get_trend(cafe_v.chunk({'st_ocean':-1}).rename({'lon_x':'lon','lat_x':'lat'}).mean('st_ocean').sel(time=slice(sday,eday))).sel(parameter=0),sb=528,title='V Cafe')


    #plot_pacific_subplot(upwelling.sel(time=slice(sday,eday)).mean(dim='time'),sb=529,title='Upwelling Cafe')
    #plot_pacific_subplot(xarray_get_trend(upwelling.sel(time=slice(sday,eday))).sel(parameter=0),sb=[5,2,10],title='Upwelling Cafe')


# + tags=[]

#deseasonaliser(xarray_detrend(cafe_co2,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
#deseasonaliser(xarray_detrend(cafe_co2_natural,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='purple')
#deseasonaliser(xarray_detrend(land_co2,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')


# + [markdown] tags=[]
# ## Interannual Variability


# + tags=[]
# Interannual Variability?
plt.figure(figsize=(20,12))
plt.subplot(311)
deseasonaliser(xarray_detrend(cafe_sst.sel(lat=slice(-10,10),time=slice('2000','2020')),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
#deseasonaliser(xarray_detrend(sst_cafe.sel(ensemble=25),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2,linestyle=':')
deseasonaliser(xarray_detrend(rey_sst.sel(lat=slice(-10,10),time=slice('2000','2020')),keep_intercept_values=False)).mean(dim=['lat','lon']).sst.plot(c='b',linewidth=2,linestyle=':')
plt.legend(['CAFE ens 25','Reynolds OISST'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('SST Interannual Variability')

plt.subplot(312)
deseasonaliser(xarray_detrend(cafe_co2.sel(lat=slice(-10,10),time=slice('2000','2020'))*60*60*-24,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
deseasonaliser(xarray_detrend(cafe_co2_natural.sel(lat=slice(-10,10),time=slice('2000','2020'))*60*60*-24,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='purple')
deseasonaliser(xarray_detrend(land_co2.sel(lat=slice(-10,10),time=slice('2000','2020')),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
deseasonaliser(xarray_detrend(co2_rodenbeck.sel(lat=slice(-10,10),time=slice('2000','2020'))*-1,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='g',linewidth=2,linestyle=':')
plt.legend(['Cafe CO2','Cafe CO2 natural','Landschutszer CO2','Rodenbeck CO2'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.ylim([-1.5,1.5])
plt.title('CO2 Interannual Variability')

plt.subplot(313)
#deseasonaliser(xarray_detrend(npp_cafe_25,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
#deseasonaliser(xarray_detrend(npp_cafe_25_sed,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linestyle=':',linewidth=2)
deseasonaliser(xarray_detrend(cafe_pprod.sel(lat=slice(-10,10),time=slice('2000','2020'))*60*60*24*6.625,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='g',linewidth=2)
deseasonaliser(xarray_detrend(npp.sel(lat=slice(-10,10),time=slice('2000','2020'))/12,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
plt.legend(['CAFE Gross Primary Prod','Satellite NPP'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('NPP Interannual Variability')
plt.tight_layout()


# + tags=[]
#(cafe_pprod.sel(lat=slice(-10,10),time=slice('2000','2020'))*60*60*24*6.625).mean(dim=['lat','lon']).plot(c='g',linewidth=2)
#(npp.sel(lat=slice(-10,10),time=slice('2000','2020'))/12).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
#plt.legend(['Cafe GPP','Satellite NPP'])


# + tags=[]
(cafe_sst.sel(lat=slice(-10,10),time=slice('2000','2020'))).mean(dim=['lat','lon']).groupby('time.month').mean().plot(c='k',linewidth=2)


# + tags=[]
#((cafe_co2.sel(lat=slice(-10,10),time=slice('2000','2020'))*60*60*-24)).mean(dim=['lat','lon']).groupby('time.month').mean().plot()


# + tags=[]
# Seasonal Variability? 
plt.figure(figsize=(20,12))
plt.subplot(311)
((cafe_sst.sel(lat=slice(-10,10),time=slice('2000','2020')))).mean(dim=['lat','lon']).groupby('time.month').mean().plot(c='k',linewidth=2)
#deseasonaliser(xarray_detrend(sst_cafe.sel(ensemble=25),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2,linestyle=':')
((rey_sst.sel(lat=slice(-10,10),time=slice('2000','2020')))).mean(dim=['lat','lon']).sst.groupby('time.month').mean().plot(c='b',linewidth=2,linestyle=':')
plt.legend(['CAFE ens 25','Reynolds OISST'])
#plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('SST Seasonality')

plt.subplot(312)
((cafe_co2.sel(lat=slice(-10,10),time=slice('2000','2020'))*60*60*-24)).mean(dim=['lat','lon']).groupby('time.month').mean().plot(c='k',linewidth=2)
((cafe_co2_natural.sel(lat=slice(-10,10),time=slice('2000','2020'))*60*60*-24)).mean(dim=['lat','lon']).groupby('time.month').mean().plot(c='purple')
((land_co2.sel(lat=slice(-10,10),time=slice('2000','2020')))).mean(dim=['lat','lon']).groupby('time.month').mean().plot(c='b',linewidth=2,linestyle=':')
((co2_rodenbeck.sel(lat=slice(-10,10),time=slice('2000','2020')))).mean(dim=['lat','lon']).groupby('time.month').mean().plot(c='g',linewidth=2,linestyle=':')
plt.legend(['Cafe CO2','Cafe CO2 natural','Landschutszer CO2','Rodenbeck CO2'])
#plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
#plt.ylim([-1.5,1.5])
plt.title('CO2 Seasonality')

plt.subplot(313)
#deseasonaliser(xarray_detrend(npp_cafe_25,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
#deseasonaliser(xarray_detrend(npp_cafe_25_sed,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linestyle=':',linewidth=2)
((cafe_pprod.sel(lat=slice(-10,10),time=slice('2000','2020'))*60*60*24*6.625)).mean(dim=['lat','lon']).groupby('time.month').mean().plot(c='g',linewidth=2)
((npp.sel(lat=slice(-10,10),time=slice('2000','2020'))/12)).mean(dim=['lat','lon']).groupby('time.month').mean().plot(c='b',linewidth=2,linestyle=':')
plt.legend(['CAFE Gross Primary Prod','Satellite NPP'])
#plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('NPP Seasonality')
plt.tight_layout()


# + [markdown] tags=[]
# # Figure 2: EUC Overview


# + tags=[]
contour_level=0.4
plt.figure(figsize=(18,4))


plt.subplot(141)
cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe80.collections[0].set_label('1980-1990')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC 1980-1990')


plt.subplot(142)
cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe10=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contour(levels=[contour_level],colors='m')
plt.xlim([150,270])
plt.ylim([-300,0])
cafe10.collections[0].set_label('2010-2020')
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC 2010-2020')



plt.subplot(143)
xarray_get_trend(cafe_u.sel(lat_x=0,method='nearest')).sel(parameter=0).plot()
#cafe_u.sel(lat_x=0,method='nearest').mean(dim='time').plot.contour(levels=[contour_level])
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe10=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contour(levels=[contour_level],colors='m')

cafe80.collections[0].set_label('1980-1990')
cafe10.collections[0].set_label('2010-2020')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC Trend')


plt.subplot(144)
euc_obs=obs_current.U_320.sel(lat=slice(-3,3)).mean(dim='lat').interpolate_na('depth').interpolate_na('lon')#.mean(dim=['time']).plot()
euc_obs['depth']=euc_obs['depth']*-1
euc_obs.sel(time=slice('1980','2020')).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))

obs80=euc_obs.sel(time=slice('1980','2020')).mean(dim='time').plot.contour(levels=[contour_level],colors='k',label='Observations (1980-2020)')
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe10=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contour(levels=[contour_level],colors='m')

obs80.collections[0].set_label('Observations (1980-2020)')
cafe80.collections[0].set_label('1980-1990')
cafe10.collections[0].set_label('2010-2020')
plt.legend(loc='lower right')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.title('TAO-TRITON mean')
plt.tight_layout()


# + [markdown] tags=[]
# How about including ENSO breakdown?


# + tags=[]



# + tags=[]
contour_level=0.4
plt.figure(figsize=(18,4))


plt.subplot(141)
cafe_u.sel(lat_x=0,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe80.collections[0].set_label('CP')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC CP')



plt.subplot(142)
cafe_u.sel(lat_x=0,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe80.collections[0].set_label('EP')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC EP')


plt.subplot(143)
cafe_u.sel(lat_x=0,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe80.collections[0].set_label('Nina')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC Nina')

plt.subplot(144)
cafe_u.sel(lat_x=0,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe80.collections[0].set_label('Neutral')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC Neutral')




# + tags=[]



# + tags=[]



# + tags=[]
#euc_obs.sel(time=slice('2000','2025')).mean(dim='time').plot.contourf()


# + tags=[]
plt.figure(figsize=(15,7))
plt.subplot(251)
levs=np.arange(-0.08,0.081,0.01)

cafe_v.sel(lat_x=-9,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.title('Meridonal Currents at 9S')
plt.xlim([150,270])

plt.subplot(252)
cafe_v.sel(lat_x=-7,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.title('Meridonal Currents at 7S')
plt.xlim([150,270])

plt.subplot(253)
cafe_v.sel(lat_x=-5,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal Currents at 5S')

plt.subplot(254)
cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal Currents at 3S')

plt.subplot(255)
cafe_v.sel(lat_x=-1,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal Currents at 1S')

plt.subplot(256)
cafe_v.sel(lat_x=9,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal Currents at 9N')

plt.subplot(257)
cafe_v.sel(lat_x=7,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal Currents at 7N')

plt.subplot(258)
cafe_v.sel(lat_x=5,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal Currents at 5N')

plt.subplot(259)
cafe_v.sel(lat_x=3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal Currents at 3N')

plt.subplot(2,5,10)
cafe_v.sel(lat_x=1,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal Currents at 1N')

plt.tight_layout()




#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
# -






dic_tx_v.sel(lat=-9,method='nearest')

# + tags=[]



# + tags=[]
def get_area(dat,cs):
    #https://stackoverflow.com/a/70710241/9965678
    areaHold=[]
    for i in range(len(cs.collections)):
        cont = cs.collections[i]
        vs = cont.get_paths()
        contour_data=0
        for contour in vs:
            x=contour.vertices[:,0]
            y=contour.vertices[:,1]
            contour_data+=dat.sel(st_ocean=np.unique(y),lon=np.unique(x),method='nearest').sum().values#.drop_duplicates('lon')#.drop_duplicates('st_ocean')#.mean().values
            print(contour_data)#.mean())
            #area+=0.5*np.mean(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
            #print(len(vs))
            #print(area)
            #area=np.abs(area)
        areaHold.append(contour_data)#/len(vs))
    return areaHold


# + tags=[]
def xarray_posneg_calc(ds,cutoff=0,mean=True,printer=True):
    pos=ds.where(ds>cutoff)
    neg=ds.where(ds<-cutoff)
    if mean==True:
        res=[pos.mean().values,neg.mean().values]
    elif mean==False:
        res=[pos.sum().values,neg.sum().values]
    if printer==True:
        print(np.array(res))
    return np.array(res)

# + tags=[]
# NEW TX TEST


#adic_utx adic_vtx dic_vtx dic_utx

plt.figure(figsize=(20,7))
plt.subplot(251)
levs=None#np.arange(-150,175,25)
contour=[0]#60,-60]
lat=3
plt.subplot(241)
adic_vtx.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
adic_vtx.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(adic_vtx.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
#print(asum)
plt.title('Neutral Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
adic_vtx.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
adic_vtx.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(adic_vtx.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
#print(asum)
plt.title('EP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
adic_vtx.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
adic_vtx.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(adic_vtx.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])


plt.subplot(244)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(adic_vtx.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])


lat=-3
plt.subplot(245)
adic_vtx.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
adic_vtx.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(adic_vtx.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])

plt.subplot(246)
adic_vtx.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
adic_vtx.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(adic_vtx.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(247)
adic_vtx.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
adic_vtx.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(adic_vtx.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(248)
adic_vtx.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
cs=adic_vtx.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(adic_vtx.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])



plt.tight_layout()

# Mmol DIC/m2/s


#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]



# + tags=[]



# + tags=[]
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-150,175,25)
contour=[0]#60,-60]
lat=3
plt.subplot(241)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).    mean(dim='time'))
#print(asum)
plt.title('Neutral Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
#print(asum)
plt.title('EP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])


plt.subplot(244)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])


lat=-3
plt.subplot(245)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])

plt.subplot(246)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(247)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(248)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
cs=dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])



plt.tight_layout()

# Mmol DIC/m2/s


#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
#(asum1[0]
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-150,175,25)
contour=[0]#60,-60]
lat=3
plt.subplot(241)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
#print(asum)
plt.title('Neutral Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
#print(asum)
plt.title('EP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])


plt.subplot(244)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])


lat=-3
plt.subplot(245)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])

plt.subplot(246)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(247)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(248)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
cs=dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])



plt.tight_layout()

# Mmol DIC/m2/s


#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))/1000)*12


# + tags=[]
#(cafe_potential_density).sel(lat=3,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf()
(cafe_potential_density).sel(lat=3,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3])


# + tags=[]



# + tags=[]
#anth DIC MEAN
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-2.5,2.75,0.25)
contour=[0]#60,-60]
lat=3
plt.subplot(241)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
#density.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf()
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    
#print(asum)
plt.title('Neutral Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

#print(asum)
plt.title('EP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(244)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

lat=-3
plt.subplot(245)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(246)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(247)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(248)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.tight_layout()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
#anth DIC EUC CROSS SECTIONS 

levs=np.arange(-30,32,2)
contour=[0]#60,-60]
lon=180
lslice=slice(-3,3)

lons=[160,250]
for lon in lons:
    plt.figure(figsize=(20,4))
    #plt.subplot(251)
    plt.subplot(141)
    anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=neutral_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
    anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=neutral_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
    asum=xarray_posneg_calc(anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=neutral_events,lat=lslice).mean(dim='time'))
    
    (cafe_potential_density).sel(lon=lon,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    #print(asum)
    plt.title(f'Neutral Zonal DIC at 0N, {lon}')
    #plt.xlim([150,270])
    plt.ylim([-300,0])
    plt.xlim([-3,3])

    plt.subplot(142)
    anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=ep_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
    anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=ep_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
    asum=xarray_posneg_calc(anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=ep_events,lat=lslice).mean(dim='time'))
    (cafe_potential_density).sel(lon=lon,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    #print(asum)
    plt.title(f'EP Zonal DIC at 0N, {lon}')
    #plt.xlim([150,270])
    plt.ylim([-300,0])
    plt.xlim([-3,3])

    plt.subplot(143)
    anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=cp_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
    anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=cp_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
    asum=xarray_posneg_calc(anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=cp_events,lat=lslice).mean(dim='time'))
    (cafe_potential_density).sel(lon=lon,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    plt.title(f'CP Zonal DIC at 0N, {lon}')
    #plt.xlim([150,270])
    plt.ylim([-300,0])
    plt.xlim([-3,3])

    plt.subplot(144)
    anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=nina_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
    anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=nina_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
    asum=xarray_posneg_calc(anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=nina_events,lat=lslice).mean(dim='time'))
    (cafe_potential_density).sel(lon=lon,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    plt.title(f'Nina Zonal DIC at 0N, {lon}')
    #plt.xlim([150,270])
    plt.ylim([-300,0])
    plt.xlim([-3,3])
    
    
    plt.tight_layout()
    plt.show()
    # Mmol DIC/m2/s
    #cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
d1=1023
d2=1024.5
d3=1026.3


# + tags=[]
1023


# + [markdown] tags=[]
# ## Start Estimating the ENSO transport through EUC
#
# Get sum area integration for AnthDIC flow in molC/s through each of the sides of the box.


# + tags=[]
def xarray_posneg_calc(ds,cutoff=0,mean=False,printer=True):
    pos=ds.where(ds>cutoff)
    neg=ds.where(ds<-cutoff)
    if mean==True:
        res=[pos.mean().values,neg.mean().values]
    elif mean==False:
        res=[pos.sum().values,neg.sum().values]
    if printer==True:
        #print(np.array(res))
        #print((res[0]/-grid_multiplier_lon.sum().values,(res[1]/-grid_multiplier_lon.sum().values)))
        # PRINTING Tonne/s
        # assumin mmol/s input 
        print(np.round(((res[0]/1000/1000)*12),0),#*((grid_multiplier_lon.T*-1).sum().values),
              np.round(((res[1]/1000/1000)*12),0),
              np.round((abs(res[0]/1000/1000)*12)-abs((res[1]/1000/1000)*12),0))#*(grid_multiplier_lon.T*-1).sum().values))
        
    return np.array(res)


# + tags=[]
grid_multiplier_lon=grid_rg.depth_lon_m2
grid_multiplier_lon=grid_multiplier_lon.rename({'lon_m':'lon','depth_diff':'st_ocean'})
grid_multiplier_lon['lon']=grid_rg.lon
grid_multiplier_lon['st_ocean']=grid_rg.depth.values

grid_multiplier_lat=grid_rg.depth_lat_m2
grid_multiplier_lat=grid_multiplier_lat.rename({'lat_m':'lat','depth_diff':'st_ocean'})
grid_multiplier_lat['lat']=grid_rg.lat
grid_multiplier_lat['st_ocean']=grid_rg.depth.values



# + tags=[]
#grid_multiplier_lat
#grid_multiplier_lon


# + tags=[]



# + tags=[]
#grid_lat_multiplier=grid_rg.sel(lat=3,method='nearest')[['depth_m2']]
#grid_lat_multiplier=grid_lat_multiplier.rename({'depth_diff':'st_ocean'})
#grid_lat_multiplier['st_ocean']=grid.depth
#grid_lat_multiplier


# + tags=[]
# Density Check
(cafe_potential_density).sel(lon=slice(160,250)).sel(time=nina_events).mean(dim=['time','lon']).plot.contourf(levels=np.arange(1020,1030,1))#(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
(cafe_potential_density).sel(lon=slice(160,250)).sel(time=nina_events).mean(dim=['time','lon']).plot.contour(levels=[1023,1024.5,1026.3],colors='k',linestyles=":",linewidths=3)


# + tags=[]
#grid_multiplier.depth_m2#.depth_m2#.coords


# + tags=[]



# + tags=[]
asum[0]/-grid_multiplier_lon.sum()


# + tags=[]
grid#*grid_multiplier_lon


# + tags=[]
((grid_multiplier_lon*-1).T).plot()


# + tags=[]



# + tags=[]
lat=-3
adic_tx_n=(grid_multiplier_lon*anth_dic_tx_v.sel(lat=lat,method='nearest').sel(st_ocean=slice(0,-300),lon=slice(160,250))).sel(time=neutral_events).mean(dim='time')
# So this is 

adic_tx_n.T.plot()
plt.show()
dat=(adic_tx_n.T)
dat.plot()
asum=xarray_posneg_calc(dat)


# + tags=[]



# + tags=[]
(dat.where(dat>0).sum())#(grid_multiplier_lon*-1)).mean()


# + tags=[]
#anth DIC with GRID???
# SIde Budget


#grid_multiplier_lat
#grid_multiplier_lon

plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-1e7,1e7,1e6)
contour=[0]#60,-60]
lat=3

adic_tx_n=((grid_multiplier_lon*(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(st_ocean=slice(0,-500)))).T)/1000000

plt.subplot(241)        
adic_tx_n.sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_n.sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_n.sel(time=neutral_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    
#print(asum)
plt.title('Neutral Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
adic_tx_n.sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_n.sel(time=ep_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_n.sel(time=ep_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

#print(asum)
plt.title('EP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
adic_tx_n.sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_n.sel(time=cp_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_n.sel(time=cp_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(244)
adic_tx_n.sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_n.sel(time=nina_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_n.sel(time=nina_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

lat=-3
adic_tx_s=(grid_multiplier_lon*(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(st_ocean=slice(0,-500)))).T#.load()

plt.subplot(245)
adic_tx_s.sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_s.sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_s.sel(time=neutral_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(246)
adic_tx_s.sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_s.sel(time=ep_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_s.sel(time=ep_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(247)
adic_tx_s.sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_s.sel(time=cp_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_s.sel(time=cp_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(248)
adic_tx_s.sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_s.sel(time=nina_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_s.sel(time=nina_events).mean(dim='time'))

(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.tight_layout()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
#anth DIC with GRID???
# SIde Budget


#grid_multiplier_lat
#grid_multiplier_lon

plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-1e7,1e7,1e6)
contour=[0]#60,-60]
lat=3

adic_tx_n=(grid_multiplier_lon*(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(st_ocean=slice(0,-500)))).T

plt.subplot(241)        
adic_tx_n.sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_n.sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_n.sel(time=neutral_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    
#print(asum)
plt.title('Neutral Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
adic_tx_n.sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_n.sel(time=ep_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_n.sel(time=ep_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

#print(asum)
plt.title('EP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
adic_tx_n.sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_n.sel(time=cp_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_n.sel(time=cp_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(244)
adic_tx_n.sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_n.sel(time=nina_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_n.sel(time=nina_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

lat=-3
adic_tx_s=(grid_multiplier_lon*(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(st_ocean=slice(0,-500)))).T#.load()

plt.subplot(245)
adic_tx_s.sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_s.sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_s.sel(time=neutral_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(246)
adic_tx_s.sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_s.sel(time=ep_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_s.sel(time=ep_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(247)
adic_tx_s.sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_s.sel(time=cp_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_s.sel(time=cp_events).mean(dim='time'))
(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(248)
adic_tx_s.sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
adic_tx_s.sel(time=nina_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(adic_tx_s.sel(time=nina_events).mean(dim='time'))

(cafe_potential_density).sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)

#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.tight_layout()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]



# + tags=[]
#anth DIC

levs=np.arange(-1e7,1e7,1e6)
contour=[0]#60,-60]
lon=180
lslice=slice(-3,3)

lons=[160,250]
for lon in lons:
    plt.figure(figsize=(20,4))
    #plt.subplot(251)
    plt.subplot(141)
    
    adic_tx_b=grid_multiplier_lat.T*anth_dic_tx_u.sel(lon=lon,method='nearest').sel(lat=lslice)#.mean(dim='time')
    
    #(grid_multiplier_lon*(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(st_ocean=slice(0,-500)))).T

    adic_tx_b.sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
    adic_tx_b.sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour,colors='k')
    asum=xarray_posneg_calc(adic_tx_b.sel(time=neutral_events).mean(dim='time'))
    
    (cafe_potential_density).sel(lon=lon,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    #print(asum)
    plt.title(f'Neutral Zonal DIC at 0N, {lon}')
    #plt.xlim([150,270])
    plt.ylim([-300,0])
    plt.xlim([-3,3])

    plt.subplot(142)
    adic_tx_b.sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
    adic_tx_b.sel(time=ep_events).mean(dim='time').plot.contour(levels=contour,colors='k')
    asum=xarray_posneg_calc(adic_tx_b.sel(time=ep_events).mean(dim='time'))
    (cafe_potential_density).sel(lon=lon,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    #print(asum)
    plt.title(f'EP Zonal DIC at 0N, {lon}')
    #plt.xlim([150,270])
    plt.ylim([-300,0])
    plt.xlim([-3,3])

    plt.subplot(143)
    adic_tx_b.sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
    adic_tx_b.sel(time=cp_events).mean(dim='time').plot.contour(levels=contour,colors='k')
    asum=xarray_posneg_calc(adic_tx_b.sel(time=cp_events).mean(dim='time'))
    (cafe_potential_density).sel(lon=lon,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    plt.title(f'CP Zonal DIC at 0N, {lon}')
    #plt.xlim([150,270])
    plt.ylim([-300,0])
    plt.xlim([-3,3])

    plt.subplot(144)
    adic_tx_b.sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
    adic_tx_b.sel(time=nina_events).mean(dim='time').plot.contour(levels=contour,colors='k')
    asum=xarray_posneg_calc(adic_tx_b.sel(time=nina_events).mean(dim='time'))
    (cafe_potential_density).sel(lon=lon,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
    plt.title(f'Nina Zonal DIC at 0N, {lon}')
    #plt.xlim([150,270])
    plt.ylim([-300,0])
    plt.xlim([-3,3])
    
    
    plt.tight_layout()
    plt.show()
    # Mmol DIC/m2/s
    #cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]



# + tags=[]



# + tags=[]



# + tags=[]
#anth DIC

levs=np.arange(-160000,20000,20000)
contour=[0]#60,-60]
lon=180
lslice=slice(-3,3)

#lons=[160,250]
#for lon in lons:
plt.figure(figsize=(20,4))
#plt.subplot(251)
plt.subplot(141)

aco2_flux=-cafe_co2_anth.sel(lat=lslice,lon=slice(150,270))*grid_rg.m2

#(grid_multiplier_lon*(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(st_ocean=slice(0,-500)))).T

aco2_flux.sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
aco2_flux.sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(aco2_flux.sel(time=neutral_events).mean(dim='time'))

#(cafe_potential_density).sel(lon=lon,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
#print(asum)
plt.title(f'Neutral Anth CO2 flux')
#plt.xlim([150,270])
plt.xlim([150,270])
plt.ylim([-3,3])

plt.subplot(142)
aco2_flux.sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
aco2_flux.sel(time=ep_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(aco2_flux.sel(time=ep_events).mean(dim='time'))
#(cafe_potential_density).sel(lon=lon,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
#print(asum)
plt.title(f'EP Anth CO2 flux')
#plt.xlim([150,270])
plt.xlim([150,270])
plt.ylim([-3,3])

plt.subplot(143)
aco2_flux.sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
aco2_flux.sel(time=cp_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(aco2_flux.sel(time=cp_events).mean(dim='time'))
#(cafe_potential_density).sel(lon=lon,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
plt.title(f'CP Anth CO2 flux')
#plt.xlim([150,270])
plt.xlim([150,270])
plt.ylim([-3,3])

plt.subplot(144)
aco2_flux.sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
aco2_flux.sel(time=nina_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(aco2_flux.sel(time=nina_events).mean(dim='time'))
#(cafe_potential_density).sel(lon=lon,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
plt.title(f'Nina Anth CO2 flux')
#plt.xlim([150,270])
plt.xlim([150,270])
#plt.ylim([-3,3])


plt.tight_layout()
plt.show()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
#anth DIC UPWELLING
levs=None#np.arange(-1e7,1e7,1e6)
contour=[0]#60,-60]
lon=180
lslice=slice(-3,3)

#lons=[160,250]
#for lon in lons:
plt.figure(figsize=(20,4))
#plt.subplot(251)
plt.subplot(141)

#aco2_flux=-cafe_co2_anth.sel(lat=lslice,lon=slice(150,270))
upwelling_dat=anth_dic_tx_w.sel(st_ocean=300,method='nearest').sel(lat=lslice,lon=slice(150,270))*grid_rg.m2
#co2_flux=-cafe_co2_anth.sel(lat=lslice,lon=slice(150,270))*grid_rg.m2
#(grid_multiplier_lon*(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(st_ocean=slice(0,-500)))).T

upwelling_dat.sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
upwelling_dat.sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(upwelling_dat.sel(time=neutral_events).mean(dim='time'))

#(cafe_potential_density).sel(lon=lon,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
#print(asum)
plt.title(f'Neutral Anth DIC upwelling')
#plt.xlim([150,270])
plt.xlim([150,270])
plt.ylim([-3,3])

plt.subplot(142)
upwelling_dat.sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
upwelling_dat.sel(time=ep_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(upwelling_dat.sel(time=ep_events).mean(dim='time'))
#(cafe_potential_density).sel(lon=lon,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
#print(asum)
plt.title(f'EP Anth DIC upwelling')
#plt.xlim([150,270])
plt.xlim([150,270])
plt.ylim([-3,3])

plt.subplot(143)
upwelling_dat.sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
upwelling_dat.sel(time=cp_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(upwelling_dat.sel(time=cp_events).mean(dim='time'))
#(cafe_potential_density).sel(lon=lon,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
plt.title(f'CP Anth DIC upwelling')
#plt.xlim([150,270])
plt.xlim([150,270])
plt.ylim([-3,3])

plt.subplot(144)
upwelling_dat.sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
upwelling_dat.sel(time=nina_events).mean(dim='time').plot.contour(levels=contour,colors='k')
asum=xarray_posneg_calc(upwelling_dat.sel(time=nina_events).mean(dim='time'))
#(cafe_potential_density).sel(lon=lon,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=[1023,1024.5,1026.3],colors='gray',linestyles=":",linewidths=3)
plt.title(f'Nina Anth DIC upwelling')
#plt.xlim([150,270])
plt.xlim([150,270])
plt.ylim([-3,3])


plt.tight_layout()
plt.show()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]



# + [markdown] tags=[]
# # END OF BOX ESTIMATE?


# + tags=[]



# + tags=[]



# + tags=[]



# + tags=[]



# + tags=[]



# + tags=[]



# + tags=[]



# + tags=[]
# Total CO2 Into Ocean
plt.figure(figsize=(20,4))
#plt.subplot(251)
contour=[0]
lslice=slice(-3,3)
levs=None#np.arange(-0.01,0.01,0.001)
plt.subplot(141)
(cafe_co2*-1).sel(time=neutral_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
(cafe_co2*-1).sel(time=neutral_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc((cafe_co2*-1).sel(time=neutral_events,lat=lslice).mean(dim='time'))
#print(asum)
plt.title(f'Neutral CO2 flux')
#plt.xlim([150,270])
#plt.ylim([-300,0])
plt.ylim([-3,3])

plt.subplot(142)
(cafe_co2*-1).sel(time=ep_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
(cafe_co2*-1).sel(time=ep_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc((cafe_co2*-1).sel(time=ep_events,lat=lslice).mean(dim='time'))
#print(asum)
plt.title(f'EP CO2 flux')

plt.ylim([-3,3])
#plt.xlim([150,270])
#plt.ylim([-300,0])


plt.subplot(143)
(cafe_co2*-1).sel(time=cp_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
(cafe_co2*-1).sel(time=cp_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc((cafe_co2*-1).sel(time=cp_events,lat=lslice).mean(dim='time'))
plt.title(f'CP CO2 flux')
plt.ylim([-3,3])
#plt.xlim([150,270])
#plt.ylim([-300,0])

plt.subplot(144)
(cafe_co2*-1).sel(time=nina_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
(cafe_co2*-1).sel(time=nina_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc((cafe_co2*-1).sel(time=nina_events,lat=lslice).mean(dim='time'))
plt.title(f'Nina CO2 flux')
plt.ylim([-3,3])
#plt.xlim([150,270])
#plt.ylim([-300,0])
plt.tight_layout()
plt.show()


# + tags=[]



# + tags=[]



# + tags=[]
#.sel(time=slice('2000-01-01','2020-01-01')


# + tags=[]



# + tags=[]
grid.m2


# + tags=[]



# + tags=[]
grid.m2


# + tags=[]
rg_m


# + tags=[]
upwelling_dat#*grid.m2


# + tags=[]
# Anth DIC 300M Upwelling

upwelling_dat=anth_dic_tx_w.sel(st_ocean=300,method='nearest')*rg_m
#upwelling_dat=upwelling_dat*grid.m2
#lon
plt.figure(figsize=(20,4))
#plt.subplot(251)
contour=[0]
lslice=slice(-3,3)
levs=None#np.arange(-0.01,0.01,0.001)
plt.subplot(141)
upwelling_dat.sel(time=neutral_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
upwelling_dat.sel(time=neutral_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(upwelling_dat.sel(time=neutral_events,lat=lslice).mean(dim='time'))
#print(asum)
plt.title(f'Neutral Anth Upwelling flux')
#plt.xlim([150,270])
#plt.ylim([-300,0])
#plt.ylim([-3,3])

plt.subplot(142)
upwelling_dat.sel(time=ep_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
upwelling_dat.sel(time=ep_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(upwelling_dat.sel(time=ep_events,lat=lslice).mean(dim='time'))
#print(asum)
plt.title(f'EP Anth Upwelling flux')

#plt.ylim([-3,3])
#plt.xlim([150,270])
#plt.ylim([-300,0])


plt.subplot(143)
upwelling_dat.sel(time=cp_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
upwelling_dat.sel(time=cp_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(upwelling_dat.sel(time=cp_events,lat=lslice).mean(dim='time'))
plt.title(f'CP Anth Upwelling flux')
#plt.ylim([-3,3])
#plt.xlim([150,270])
#plt.ylim([-300,0])

plt.subplot(144)
upwelling_dat.sel(time=nina_events,lat=lslice).mean(dim='time').plot.contourf(levels=levs)
upwelling_dat.sel(time=nina_events,lat=lslice).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(upwelling_dat.sel(time=nina_events,lat=lslice).mean(dim='time'))
plt.title(f'Nina Anth Upwelling flux')
#plt.ylim([-3,3])
#plt.xlim([150,270])
#plt.ylim([-300,0])
plt.tight_layout()
plt.show()


# + tags=[]



# + tags=[]
plt.subplot(243)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])


plt.subplot(244)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])


lat=-3
plt.subplot(245)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])

plt.subplot(246)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(247)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(248)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])


plt.tight_layout()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
#anth DIC
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-2.5,2.75,0.25)
contour=[0]#60,-60]
lat=15
plt.subplot(241)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
#print(asum)
plt.title('Neutral Meridonal DIC at 9N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
#print(asum)
plt.title('EP Meridonal DIC at 9N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 9N')
plt.xlim([150,270])


plt.subplot(244)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Meridonal DIC at 9N')
plt.xlim([150,270])


lat=-15
plt.subplot(245)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
plt.title('Neutral Meridonal DIC at 9S')
plt.xlim([150,270])

plt.subplot(246)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
plt.title('EP Meridonal DIC at 9S')
plt.xlim([150,270])


plt.subplot(247)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 9S')
plt.xlim([150,270])


plt.subplot(248)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 9S')
plt.xlim([150,270])


plt.tight_layout()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
dic_tx_u.sel(lat=lat,lon=180,method='nearest').plot()#.sel(time=neutral_events).mean(dim='time').plot()


# + tags=[]
dic_tx_v.sel(lat=lat,lon=180,method='nearest').plot()#.sel(time=neutral_events).mean(dim='time').plot()


# + tags=[]
#anth DIC
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-25,27.5,2.5)
contour=[0]#60,-60]
lat=0
plt.subplot(141)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
#print(asum)
plt.title('Neutral Zonal DIC at 0N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(142)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
#print(asum)
plt.title('EP Zonal DIC at 0N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(143)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Zonal DIC at 0N')
plt.xlim([150,270])


plt.subplot(144)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Zonal DIC at 0N')
plt.xlim([150,270])

plt.tight_layout()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
anth_dic_tx_u.sel(lon=lon,method='nearest').sel(time=neutral_events).mean(dim='time').plot(levels=np.arange(-30,30,1))


# + tags=[]



# + tags=[]
dic_tx_u.sel(lon=172,method='nearest').mean(dim='time').plot.contourf()


# + tags=[]
dic_tx_v.sel(lon=154,method='nearest').mean(dim='time').plot.contourf()


# + tags=[]
dic_tx_u.sel(lon=149,method='nearest').mean(dim='time').plot.contourf()


# + tags=[]



# + tags=[]
dic_tx_v.mean(dim=['st_ocean','time']).plot()


# + tags=[]
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-150,155,5)
contour=[50,-50]
lat=2
plt.subplot(241)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
plt.title('Neutral Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
plt.title('EP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])


plt.subplot(244)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])


lat=-2
plt.subplot(245)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])

plt.subplot(246)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(247)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(248)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])



plt.tight_layout()

# Mmol DIC/m2/s


#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-1000,1000,5)
contour=[0]#500,-500]
lat=0

plt.subplot(241)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
plt.title('Neutral Zonal DIC at 0N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(242)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
plt.title('EP Zonal DIC at 0N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(243)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
plt.title('CP Zonal DIC at 0N')
plt.xlim([150,270])


plt.subplot(244)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
plt.title('Nina Meridonal DIC at 0N')
plt.xlim([150,270])


lat=-3
plt.subplot(245)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
plt.title('Neutral Meridonal DIC at 9S')
plt.xlim([150,270])

plt.subplot(246)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
plt.title('EP Meridonal DIC at 9S')
plt.xlim([150,270])


plt.subplot(247)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
plt.title('CP Meridonal DIC at 9S')
plt.xlim([150,270])


plt.subplot(248)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
plt.title('Nina Meridonal DIC at 9S')
plt.xlim([150,270])



plt.tight_layout()

# Mmol DIC/m2/s


#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
# Merge
#anth DIC
plt.figure(figsize=(20,7))

levs=np.arange(-2.5,2.75,0.25)
contour=[0]#60,-60]
lat=3
plt.subplot(341)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
#print(asum)
plt.title('Neutral Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(342)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
#print(asum)
plt.title('EP Meridonal DIC at 3N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(343)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3N')
plt.xlim([150,270])


plt.subplot(344)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Meridonal DIC at 3N')
plt.xlim([150,270])




levs=np.arange(-40,40,5)
contour=[0]#500,-500]
lat=0

plt.subplot(345)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
plt.title('Neutral Zonal DIC at 0N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(346)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
plt.title('EP Zonal DIC at 0N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(347)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Zonal DIC at 0N')
plt.xlim([150,270])


plt.subplot(348)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Meridonal DIC at 0N')
plt.xlim([150,270])




levs=np.arange(-2.5,2.75,0.25)
contour=[0]#60,-60]
lat=-3
plt.subplot(3,4,9)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
plt.title('Neutral Meridonal DIC at 3S')
plt.xlim([150,270])

plt.subplot(3,4,10)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
plt.title('EP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(3,4,11)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Meridonal DIC at 3S')
plt.xlim([150,270])


plt.subplot(3,4,12)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
#asum1=get_area(dic_tx_v.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'),cs)
plt.title('Nina Meridonal DIC at 3S')
plt.xlim([150,270])


plt.tight_layout()
# Mmol DIC/m2/s
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]



# + tags=[]
#Anth DIC
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-40,40,5)
contour=[0]#500,-500]
lat=0

plt.subplot(141)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=neutral_events).mean(dim='time'))
plt.title('Neutral Zonal DIC at 0N')
plt.xlim([150,270])
plt.ylim([-300,0])

plt.subplot(142)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=ep_events).mean(dim='time'))
plt.title('EP Zonal DIC at 0N')
plt.xlim([150,270])
plt.ylim([-300,0])


plt.subplot(143)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=cp_events).mean(dim='time'))
plt.title('CP Zonal DIC at 0N')
plt.xlim([150,270])


plt.subplot(144)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=levs)
anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time').plot.contour(levels=contour)
asum=xarray_posneg_calc(anth_dic_tx_u.sel(lat=lat,method='nearest').sel(time=nina_events).mean(dim='time'))
plt.title('Nina Meridonal DIC at 0N')
plt.xlim([150,270])




plt.tight_layout()

# Mmol DIC/m2/s


# + tags=[]
#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]



# + tags=[]
eqpac


# + tags=[]



# + tags=[]



# + tags=[]



# + tags=[]
phys_eqpac


# + tags=[]
plt.figure(figsize=(20,7))
plt.subplot(251)
levs=np.arange(-150,155,5)

dic_tx_v.sel(lat=-9,method='nearest').plot.contourf(levels=levs)
plt.title('Meridonal DIC at 9S')
plt.xlim([150,270])

plt.subplot(252)
dic_tx_v.sel(lat=-7,method='nearest').plot.contourf(levels=levs)
plt.title('Meridonal DIC at 7S')
plt.xlim([150,270])

plt.subplot(253)
dic_tx_v.sel(lat=-5,method='nearest').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal DIC at 5S')

plt.subplot(254)
dic_tx_v.sel(lat=-3,method='nearest').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal DIC at 3S')

plt.subplot(255)
dic_tx_v.sel(lat=-1,method='nearest').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal DIC at 1S')

plt.subplot(256)
dic_tx_v.sel(lat=9,method='nearest').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal DIC at 9N')

plt.subplot(257)
dic_tx_v.sel(lat=7,method='nearest').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal DIC at 7N')

plt.subplot(258)
dic_tx_v.sel(lat=5,method='nearest').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal DIC at 5N')

plt.subplot(259)
dic_tx_v.sel(lat=3,method='nearest').plot.contourf(levels=levs)
plt.xlim([150,270])
plt.title('Meridonal DIC at 3N')

plt.subplot(2,5,10)
dic_tx_v.sel(lat=1,method='nearest').plot.contourf(levels=levs)
plt.xlim([150,270])
#.title('Meridonal DIC at 1N')

plt.tight_layout()

# Mmol DIC/m2/s


#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]



# + tags=[]
u_test=cafe_u.rename({'lon_x':'lon','lat_x':'lat'}).sel(lat=0,method='nearest')


# + tags=[]
dic_test=adic_cafe.sel(lat=0,method='nearest')


# + tags=[]
u_test=u_test.sel(lon=u_test.lon[:-1])
u_test['lon']=dic_test.lon


# + tags=[]



# + tags=[]
a=u_test.mean(dim='time') # u velocity m/s mean
b=u_test.std(dim='time')*3 # u velocity m/s std
c=dic_test.mean(dim='time') # dic 
d=dic_test.std(dim='time')*3





# + tags=[]
plt.figure(figsize=(20,10))
plt.subplot(141)
(a*c).plot.contourf()
plt.title('U mean * DIC mean')

plt.subplot(142)
(a*d).plot.contourf()
plt.title('U mean * DIC std')

plt.subplot(143)
(b*c).plot.contourf()
plt.title('U std * DIC mean')

plt.subplot(144)
(b*d).plot.contourf()
plt.title('U std * DIC std')

plt.tight_layout()


# + tags=[]
c.lon.values


# + tags=[]
a.lon.values


# + tags=[]



# + tags=[]
anth_dic_cafe.sel(lat=0,method='nearest').mean(dim=['st_ocean','time'])


# + tags=[]
contour_level=0.4
plt.figure(figsize=(18,4))


plt.subplot(111)
anth_dic_cafe.sel(lat=0,method='nearest').mean(dim=['time']).plot.contourf()#(levels=np.arange(-1,1.11,0.1))
#cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
#cafe80.collections[0].set_label('CP')
#plt.xlim([150,270])
#plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC mean DIC')
plt.show()


# + tags=[]
contour_level=0.4
plt.figure(figsize=(18,4))


plt.subplot(111)
anth_dic_cafe.sel(lat=0,method='nearest').mean(dim=['st_ocean']).plot.contourf()#(levels=np.arange(-1,1.11,0.1))
#cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
#cafe80.collections[0].set_label('CP')
#plt.xlim([150,270])
#plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC DIC flow')
plt.show()


# + tags=[]
contour_level=0.4
plt.figure(figsize=(18,4))


plt.subplot(141)
anth_dic_cafe.sel(lat=0,method='nearest').mean(dim=['time']).plot.contourf()
plt.title('CAFE 60 EUC Anth DIC Mean')
plt.ylim([-300,0])

plt.subplot(142)
(xarray_get_trend(anth_dic_cafe.sel(lat=0,method='nearest')).sel(parameter=0)*365).plot.contourf(cmap='viridis',levels=np.arange(0,1,0.1))
plt.title('CAFE 60 EUC Anth DIC Trend')
plt.ylim([-300,0])

plt.subplot(143)
anth_dic_tx_u.sel(lat=0,method='nearest').mean(dim=['time']).plot.contourf()#(levels=np.arange(-1,1.11,0.1))
plt.title('CAFE 60 EUC anthDIC Transport rate Mean')
plt.ylim([-300,0])
#cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
#cafe80.collections[0].set_label('CP')
#plt.xlim([150,270])
#plt.ylim([-300,0])
#plt.legend(loc='lower right')



plt.subplot(144)
(xarray_get_trend(anth_dic_tx_u.sel(lat=0,method='nearest')).drop(['lat','ensemble']).sel(parameter=0)*365).plot.contourf()
#cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
#cafe80.collections[0].set_label('CP')
#plt.xlim([150,270])
#plt.ylim([-300,0])
#plt.legend(loc='lower right')
plt.title('CAFE 60 EUC DIC transport rate Trend')
plt.ylim([-300,0])
plt.show()


# + tags=[]
(xarray_get_trend(anth_dic_tx_u.sel(lat=0,method='nearest')).drop(['lat','ensemble']).sel(parameter=0)*365).plot.contourf()


# + tags=[]
plt.subplot(142)
cafe_u.sel(lat_x=0,method='nearest').sel(time=ep_events).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe80.collections[0].set_label('EP')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC EP')


plt.subplot(143)
cafe_u.sel(lat_x=0,method='nearest').sel(time=nina_events).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe80.collections[0].set_label('Nina')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC Nina')

plt.subplot(144)
cafe_u.sel(lat_x=0,method='nearest').sel(time=neutral_events).mean(dim='time').plot.contourf(levels=np.arange(-1,1.11,0.1))
cafe80=cafe_u.sel(lat_x=0,method='nearest').sel(time=slice('1980','1990')).mean(dim='time').plot.contour(levels=[contour_level],colors='r')
cafe80.collections[0].set_label('Neutral')
plt.xlim([150,270])
plt.ylim([-300,0])
plt.legend(loc='lower right')
plt.title('CAFE 60 EUC Neutral')


# + tags=[]



# + tags=[]



# + tags=[]



# + tags=[]
adic_cafe.mean(dim='time')


# + tags=[]
cafe_v1['st_ocean']=dic.st_ocean
cafe_v1['lat']=dic.lat
cafe_v1['lon']=dic.lon
cafe_v1['time']=dic.time


# + tags=[]



# + tags=[]
# Keeps crashing Kernal??
#dic_tx=cafe_v1*dic['adic']


# + tags=[]



# + tags=[]



# + tags=[]
#dic_tx=cafe_v1*dic['adic']


# + tags=[]



# + tags=[]



# + tags=[]
cafe_v_interp=cafe_v1.interp_like(dic['adic'])


# + tags=[]
dic_tx=dic['adic']*cafe_v


# + tags=[]
plt.figure(figsize=(20,7))
plt.subplot(251)
cafe_v.sel(lat_x=-9,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.subplot(252)
cafe_v.sel(lat_x=-7,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])
plt.subplot(253)
cafe_v.sel(lat_x=-5,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.subplot(254)
cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.subplot(255)
cafe_v.sel(lat_x=-1,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.subplot(256)
cafe_v.sel(lat_x=9,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.subplot(257)
cafe_v.sel(lat_x=7,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.subplot(258)
cafe_v.sel(lat_x=5,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.subplot(259)
cafe_v.sel(lat_x=3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.subplot(2,5,10)
cafe_v.sel(lat_x=1,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))
plt.xlim([150,270])

plt.tight_layout()




#cafe_v.sel(lat_x=-3,method='nearest').sel(time=slice('2010','2020')).mean(dim='time').plot.contourf(levels=np.arange(-0.1,0.11,0.01))


# + tags=[]
#d[0,:]


# + tags=[]
deps=cafe_u.sel(lat_x=0,method='nearest').sel(st_ocean=slice(0,-300)).dropna(dim='lon_x').idxmax(dim='st_ocean',skipna=True)#.plot()


# + tags=[]
depth=deps.sel(lon_x=lon)#


# + tags=[]
cafe_u.sel(lat_x=0,method='nearest').sel(st_ocean=slice(0,-300)).dropna(dim='lon_x').idxmax(dim='st_ocean',skipna=True).plot()


# + tags=[]
# EUC cross section and 
plt.figure(figsize=(16,5))
plt.subplot(121)
deps.mean(dim='time').plot()
plt.xlim([150,270])
plt.title('Max EUC U speed depth')

plt.subplot(122)
(xarray_get_trend(deps).sel(parameter=0)*365).plot()
plt.xlim([150,270])
plt.title('Max EUC U speed Depth Trend / Yr')


# + tags=[]
cafe_u.sel(lat_x=0,method='nearest').sel(st_ocean=slice(0,-300)).dropna(dim='lon_x').mean(dim='time').argmax(dim='st_ocean',skipna=True).plot()


# + tags=[]

cafe_u.sel(lat_x=0,lon_x=180,st_ocean=-170,method='nearest').plot()#.sel(st_ocean=slice(0,-300))


# + tags=[]
dic_cafe.sel(lat=0,lon=lon,st_ocean=depth,method='nearest')


# + tags=[]
deps


# + tags=[]
adic_cafe.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest')#.plot()


# + tags=[]
anth_dic_cafe.sel(st_ocean=


# + tags=[]
#depth


# + tags=[]
v.sel(lat=0,lon=lon,sw_ocean=depth,method='nearest').plot()#.where('sw_ocean'==depth).sel(time='2010-01-01',method='nearest').values


# + tags=[]
anth_dic_cafe.name='Anth DIC'


# + tags=[]
#anth_dic_cafe


# + tags=[]
lons=[180,200,220,240,260]
anth_dic_cafe.name='Anth DIC'
varz=[anth_dic_cafe,dic_cafe,adic_cafe,cafe_temp,cafe_u,cafe_v,cafe_wt]
fig = plt.figure(figsize=((8.27*2),11.69*2))
for i,v in enumerate(varz):
    plt.subplot(7,1,i+1)
    for i,lon in enumerate(lons):
        depth=deps.sel(lon_x=lon)#.values
        #plt.subplot(5,1,i+1)
        #dic.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest')['dic'].plot()
        try:
            #xarray_detrend(deseasonaliser(
            v.sel(lat=0,lon=lon,st_ocean=depth,method='nearest').plot(label=f'{lon} {depth.mean().values}',linewidth=2)
        except:
            pass
            try:
                v.sel(lat_x=0,lon_x=lon,st_ocean=depth,method='nearest').plot(label=f'{lon} {depth.mean().values}',linewidth=2)
            except:
                v.sel(lat=0,lon=lon,sw_ocean=depth,method='nearest').plot(label=f'{lon} {depth.mean().values}',linewidth=2)
        #plt.show()
        plt.xlim(['2000-01-01','2020-01-01'])
    plt.title(f'Cross EUC {str(v.name)}')
    plt.legend()
plt.tight_layout()
plt.show()


# + tags=[]
#Seasonality
lons=[180,200,220,240,260]
anth_dic_cafe.name='Anth DIC'
varz=[anth_dic_cafe,dic_cafe,adic_cafe,cafe_temp,cafe_u,cafe_v,cafe_wt]
fig = plt.figure(figsize=((8.27*2),11.69*2))
for i,v in enumerate(varz):
    plt.subplot(7,1,i+1)
    for i,lon in enumerate(lons):
        depth=deps.sel(lon_x=lon)#.values
        #plt.subplot(5,1,i+1)
        #dic.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest')['dic'].plot()
        try:
            #xarray_detrend(deseasonaliser(
            v.sel(lat=0,lon=lon,st_ocean=depth,method='nearest').sel(time=slice('2000-01-01','2020-01-01')).groupby('time.month').mean().plot(label=f'{lon} {depth.mean().values}',linewidth=2)
        except:
            pass
            try:
                v.sel(lat_x=0,lon_x=lon,st_ocean=depth,method='nearest').sel(time=slice('2000-01-01','2020-01-01')).groupby('time.month').mean().plot(label=f'{lon} {depth.mean().values}',linewidth=2)
            except:
                v.sel(lat=0,lon=lon,sw_ocean=depth,method='nearest').sel(time=slice('2000-01-01','2020-01-01')).groupby('time.month').mean().plot(label=f'{lon} {depth.mean().values}',linewidth=2)
        #plt.show()
        #plt.xlim(['2000-01-01','2020-01-01'])
    plt.title(f'Cross EUC {str(v.name)}')
    plt.legend()
plt.tight_layout()
plt.show()


# + tags=[]
v.sel(lat=0,lon=lon,st_ocean=depth,method='nearest').sel(time=slice('2000-01-01','2020-01-01'))


# + tags=[]
lons=[180,200,220,240,260]
anth_dic_cafe.name='Anth DIC'
varz=[anth_dic_cafe,dic_cafe,adic_cafe,cafe_temp,cafe_u,cafe_v,cafe_wt]
fig = plt.figure(figsize=((8.27*2),11.69*2))
for i,v in enumerate(varz):
    plt.subplot(7,1,i+1)
    for i,lon in enumerate(lons):
        depth=deps.sel(lon_x=lon)#.values
        #plt.subplot(5,1,i+1)
        #dic.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest')['dic'].plot()
        try:
            #xarray_detrend(deseasonaliser(
            xarray_detrend(deseasonaliser(v.sel(lat=0,lon=lon,st_ocean=depth,method='nearest').sel(time=slice('2000-01-01','2020-01-01')))).plot(label=f'{lon} {depth.mean().values}',linewidth=2)
        except:
            pass
            try:
                xarray_detrend(deseasonaliser(v.sel(lat_x=0,lon_x=lon,st_ocean=depth,method='nearest').sel(time=slice('2000-01-01','2020-01-01')))).plot(label=f'{lon} {depth.mean().values}',linewidth=2)
            except:
                xarray_detrend(deseasonaliser(v.sel(lat=0,lon=lon,sw_ocean=depth,method='nearest').sel(time=slice('2000-01-01','2020-01-01')))).plot(label=f'{lon} {depth.mean().values}',linewidth=2)
        #plt.show()
        plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
    plt.title(f'Cross EUC {str(v.name)}')
    plt.legend()
plt.tight_layout()
plt.show()


# + tags=[]
lons=[180,200,220,240,260]
varz=[dic,adic,cafe_temp,cafe_u,cafe_v,cafe_wt]

for v in varz:
    plt.figure(figsize=(15,7))
    for i,lon in enumerate(lons):
        depth=deps.sel(lon_x=lon).values
        #plt.subplot(5,1,i+1)
        #dic.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest')['dic'].plot()
        try:
            v.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest').plot(label=f'{lon} {depth.mean()}',linewidth=2)
        except:
            try:
                v.sel(lat_x=0,lon_x=lon,st_ocean=depth.mean(),method='nearest').plot(label=f'{lon} {depth.mean()}',linewidth=2)
            except:
                v.sel(lat=0,lon=lon,sw_ocean=depth.mean(),method='nearest').plot(label=f'{lon} {depth.mean()}',linewidth=2)
        #plt.show()
        plt.xlim(['2000-01-01','2020-01-01'])
    plt.title(f'Cross EUC {str(v.name)}')
    plt.legend()
    plt.tight_layout()
    plt.show()


# + tags=[]
varz=[


# + tags=[]



# + tags=[]
lons=[180,200,220,240,260]
plt.figure(figsize=(20,12))
for i,lon in enumerate(lons):
    depth=deps.sel(lon_x=lon).values
    #plt.subplot(5,1,i+1)
    #dic.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest')['dic'].plot()
    ((adic_cafe.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest'))).plot(label=f'{lon} {depth.mean()}',linewidth=2)
    #plt.show()
    #plt.xlim(['2000-01-01','2020-01-01'])
plt.legend()
plt.tight_layout()


# + tags=[]
cafe_temp


# + tags=[]
lons=[180,200,220,240,260]
plt.figure(figsize=(26,8))
for i,lon in enumerate(lons):
    depth=deps.sel(lon_x=lon).values
    #plt.subplot(5,1,i+1)
    cafe_temp.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest').plot(label=f'{lon} {depth.mean()}',linewidth=2)
    #cafe_temp.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest').plot()
    #plt.show()
    plt.xlim(['2000-01-01','2020-01-01'])
plt.legend()
plt.tight_layout()


# + tags=[]
cafe_wt


# + tags=[]
lons=[180,200,220,240,260]
plt.figure(figsize=(26,8))
for i,lon in enumerate(lons):
    depth=deps.sel(lon_x=lon).values
    #plt.subplot(5,1,i+1)
    cafe_u.sel(lat_x=0,lon_x=lon,st_ocean=depth.mean(),method='nearest').plot(label=f'{lon} {depth.mean()}',linewidth=2)
    #cafe_v.sel(lat_x=0,lon_x=lon,st_ocean=depth.mean(),method='nearest').plot(label=f'{lon} {depth.mean()}',linewidth=2)
    cafe_wt.sel(lat=0,lon=lon,sw_ocean=depth.mean(),method='nearest').plot(label=f'{lon} {depth.mean()}',linewidth=2)
    
    #cafe_temp.sel(lat=0,lon=lon,st_ocean=depth.mean(),method='nearest').plot()
    #plt.show()
    #plt.xlim(['2000-01-01','2020-01-01'])
plt.legend()
plt.tight_layout()


# + tags=[]
deps#.sel(lon_x=180,method='nearest').values


# + tags=[]
deps.plot()


# + tags=[]



# + tags=[]
# SINGLE YEAR VERSION Figure 1

sday='2000-01-01'
eday='2020-01-01'
# A3 is (11.69,16.53)
fig = plt.figure(figsize=(8.27*4,11.69*4)) #Inches Portrait
plot_pacific_subplot(cafe_sst.sel(time=slice(sday,eday)).mean(dim='time'),sb=521,title='Mean SST Cafe',levels=np.arange(18,32,1))
plot_pacific_subplot(xarray_get_trend(cafe_sst.sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=522,title='Mean SST Cafe',levels=np.arange(-0.1,0.1,0.005))

plot_pacific_subplot(rey_sst.sst.sel(time=slice(sday,eday)).mean(dim='time'),sb=523,title='Mean SST Reynolds',levels=np.arange(18,32,1))
plot_pacific_subplot(xarray_get_trend(rey_sst.sst.sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=524,title='Mean SST Reynolds',levels=np.arange(-0.1,0.1,0.005))

plot_pacific_subplot(cafe_u.chunk({'st_ocean':-1}).mean('st_ocean').rename({'lon_x':'lon','lat_x':'lat'}).sel(time=slice(sday,eday)).mean(dim='time'),sb=525,title='U Cafe')
plot_pacific_subplot(xarray_get_trend(cafe_u.chunk({'st_ocean':-1}).mean('st_ocean').rename({'lon_x':'lon','lat_x':'lat'}).sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=526,title='U Cafe')

plot_pacific_subplot(cafe_v.chunk({'st_ocean':-1}).rename({'lon_x':'lon','lat_x':'lat'}).mean('st_ocean').sel(time=slice(sday,eday)).mean(dim='time'),sb=527,title='V Cafe')
plot_pacific_subplot(xarray_get_trend(cafe_v.chunk({'st_ocean':-1}).rename({'lon_x':'lon','lat_x':'lat'}).mean('st_ocean').sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=528,title='V Cafe')


plot_pacific_subplot(upwelling_cafe.sel(time=slice(sday,eday)).mean(dim='time'),sb=529,title='Upwelling Cafe')
plot_pacific_subplot(xarray_get_trend(upwelling_cafe.sel(time=slice(sday,eday))).sel(parameter=0),sb=[5,2,10],title='Upwelling Cafe')
plt.tight_layout()


# SINGLE YEAR VERSION Figure 2


sday='2000-01-01'
eday='2020-01-01'
fig = plt.figure(figsize=(40,20))
plot_pacific_subplot((land_co2*1000).sel(time=slice(sday,eday)).mean(dim='time'),sb=521,title='Land CO2')#,levels=np.arange(18,32,1))
plot_pacific_subplot(xarray_get_trend((land_co2*1000).sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=522,title='Land CO2 Trend',)

plot_pacific_subplot(cafe_co2.sel(time=slice(sday,eday)).mean(dim='time'),sb=523,title='CAFE CO2')#,levels=np.arange(18,32,1))
plot_pacific_subplot(xarray_get_trend(cafe_co2.sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=524,title='CAFE CO2 Trend')

plot_pacific_subplot(cafe_co2_natural.sel(time=slice(sday,eday)).mean(dim='time'),sb=525,title='Natural CO2')
plot_pacific_subplot(xarray_get_trend(cafe_co2_natural.sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=526,title='Natural CO2')

plot_pacific_subplot(dic_cafe.sel(st_ocean=slice(0,-100)).mean(dim='st_ocean').sel(time=slice(sday,eday)).mean(dim='time'),sb=529,title='DIC',levels=np.arange(1800,2200,20))
plot_pacific_subplot(xarray_get_trend(dic_cafe.sel(st_ocean=slice(0,-100)).mean(dim='st_ocean').sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=[5,2,10],title='DIC',levels=np.arange(-0.025,0.025,0.0025))

plot_pacific_subplot(adic_cafe.sel(st_ocean=slice(0,-100)).mean(dim='st_ocean').sel(time=slice(sday,eday)).mean(dim='time'),sb=527,title='aDIC',levels=np.arange(1800,2200,20))
plot_pacific_subplot(xarray_get_trend(adic_cafe.sel(st_ocean=slice(0,-100)).mean(dim='st_ocean').sel(time=slice(sday,eday))).sel(parameter=0)*365,sb=528,title='aDIC',levels=np.arange(-0.025,0.025,0.0025))

plt.tight_layout()
#plot_pacific_subplot(cafe_v.chunk({'st_ocean':-1}).rename({'lon_x':'lon','lat_x':'lat'}).mean('st_ocean').sel(time=slice(sday,eday)).mean(dim='time'),sb=527,title='V Cafe')
#plot_pacific_subplot(xarray_get_trend(cafe_v.chunk({'st_ocean':-1}).rename({'lon_x':'lon','lat_x':'lat'}).mean('st_ocean').sel(time=slice(sday,eday))).sel(parameter=0),sb=528,title='V Cafe')


#plot_pacific_subplot(upwelling.sel(time=slice(sday,eday)).mean(dim='time'),sb=529,title='Upwelling Cafe')
#plot_pacific_subplot(xarray_get_trend(upwelling.sel(time=slice(sday,eday))).sel(parameter=0),sb=[5,2,10],title='Upwelling Cafe')



# + tags=[]
# DOesnt work with line for some reason. 

from matplotlib import animation
from IPython.display import HTML
deps=cafe_u.sel(lat_x=0,method='nearest').sel(st_ocean=slice(0,-300)).dropna(dim='lon_x').idxmax(dim='st_ocean',skipna=True)#.plot()
# Get a handle on the figure and the axes
fig, ax = plt.subplots(figsize=(12,6))

# Plot the initial frame. 
d=deps#u.sel(lat=0,method='nearest')#.mean(dim='time').plot.contourf(cmap='bwr')
cax = d[0,:].plot()

# Next we need to create a function that updates the values for the colormesh, as well as the title.
def animate(frame):
    cax.set_array(d[frame,:].values)
    ax.set_title("Time = " + str(d.coords['time'].values[frame])[:13])

# Finally, we use the animation module to create the animation.
ani = animation.FuncAnimation(
    fig,             # figure
    animate,         # name of the function above
    frames=len(d.time),       # Could also be iterable or list
    interval=200)     # ms between frames
HTML(ani.to_jshtml())


# + [markdown] tags=[]
# ## Figure 1: Means

# + tags=[]
fig = plt.figure(figsize=(20,12))

plot_pacific_subplot(sst_cafe.sel(ensemble=25).mean(dim='time'),sb=421,title='Mean SST Cafe',levels=np.arange(18,32,1))
plot_pacific_subplot(sst_rey.mean(dim='time'),sb=422,title='Mean SST Reynolds',levels=np.arange(18,32,1))

plot_pacific_subplot(co2_cafe.sel(ensemble=25).mean(dim='time'),sb=423,title='Mean CO2 Cafe',levels=np.arange(-12,12,1),extend='min')
plot_pacific_subplot(co2_land.mean(dim='time'),sb=424,title='Mean CO2 Landschutzer',levels=np.arange(-12,12,1))
#plot_pacific_subplot(co2_rodenbeck.mean(dim='time')*-1)
plot_pacific_subplot(npp_cafe_25.mean(dim='time'),sb=425,title='Mean NPP Cafe 25',levels=np.arange(0,30,1),cmap='viridis')
#plot_pacific_subplot(npp_cafe_23.mean(dim='time'),sb=326,title='NPP Cafe 25')
plot_pacific_subplot(np_dat.mean(dim='time'),sb=426,title='Mean NPP Obs',levels=np.arange(0,30,1),cmap='viridis')
plot_pacific_subplot(npp_cafe_25_sed.mean(dim='time'),sb=427,title='Mean NPP Cafe Sediment Export',levels=np.arange(0,30,1),cmap='viridis',extend='max')

plt.tight_layout()
plt.show()

# + tags=[]



# + [markdown] tags=[]
# ## Figure 2: Trends

# + tags=[]
fig = plt.figure(figsize=(20,12))

plot_pacific_subplot(xarray_get_trend(sst_cafe.sel(ensemble=25)).sel(parameter=0)*365,sb=321,title='Trend SST CAFE',levels=np.arange(-0.09,0.09,0.03),extend='max')
plot_pacific_subplot(xarray_get_trend(sst_rey).sel(parameter=0)*365,sb=322,title='Trend SST Reynolds',levels=np.arange(-0.09,0.09,0.03),extend='max')#,levels=np.arange(18,32,1))

plot_pacific_subplot(xarray_get_trend(co2_cafe.sel(ensemble=25)).sel(parameter=0)*365,sb=323,title='Trend CO2 Cafe',levels=np.arange(-0.2,0.2,0.02),extend='both')
plot_pacific_subplot(xarray_get_trend(co2_land).sel(parameter=0)*365,sb=324,title='Trend CO2 Landschutzer',levels=np.arange(-0.2,0.2,0.02),extend='both')
#plot_pacific_subplot(co2_rodenbeck.mean(dim='time')*-1)
plot_pacific_subplot(xarray_get_trend(npp_cafe_25).sel(parameter=0)*365,sb=325,title='Trend NPP Cafe 25')
#plot_pacific_subplot(npp_cafe_23.mean(dim='time'),sb=326,title='NPP Cafe 25')
plot_pacific_subplot(xarray_get_trend(np_dat).sel(parameter=0)*365,sb=326,title='Trend NPP Obs')
plt.tight_layout()
plt.show()


# + tags=[]



# + [markdown] tags=[]
# # Figure 3: Seasonal Magnitude

# + tags=[]
fig = plt.figure(figsize=(20,12))

plot_pacific_subplot(sst_cafe.sel(ensemble=25).groupby('time.month').mean().std(dim='month'),sb=321,title='Seasonal Amplitude SST Cafe',cmap='viridis',levels=np.arange(0,4,0.5))
plot_pacific_subplot(sst_rey.groupby('time.month').mean().std(dim='month'),sb=322,title='Seasonal Amplitude SST Reynolds',cmap='viridis',levels=np.arange(0,4,0.5))

plot_pacific_subplot(co2_cafe.sel(ensemble=25).groupby('time.month').mean().std(dim='month'),sb=323,title='Seasonal Amplitude CO2 Cafe',cmap='viridis',levels=np.arange(0,12,1))#,levels=np.arange(-12,12,1))
plot_pacific_subplot(co2_land.groupby('time.month').mean().std(dim='month'),sb=324,title='Seasonal Amplitude CO2 Landschutzer',cmap='viridis',levels=np.arange(0,12,1))#,levels=np.arange(-12,12,1))
#plot_pacific_subplot(co2_rodenbeck.mean(dim='time')*-1)
plot_pacific_subplot(npp_cafe_25.groupby('time.month').mean().std(dim='month'),sb=325,title='Seasonal Amplitude NPP Cafe 25')
#plot_pacific_subplot(npp_cafe_23.mean(dim='time'),sb=326,title='NPP Cafe 25')
plot_pacific_subplot(np_dat.groupby('time.month').mean().std(dim='month'),sb=326,title='Seasonal Amplitude NPP Obs')
plt.tight_layout()
plt.show()

# + tags=[]



# + [markdown] tags=[]
# # Figure 3b: Seasonal Magnitude / Peak Month

# + tags=[]
fig = plt.figure(figsize=(20,12))

plot_pacific_subplot(sst_cafe.sel(ensemble=25).groupby('time.month').mean().idxmax(dim='month'),sb=321,title='Peak Month  SST Cafe',cmap='viridis',levels=np.arange(1,13,1))
plot_pacific_subplot(sst_rey.groupby('time.month').mean().idxmax(dim='month'),sb=322,title='Peak Month SST Reynolds',cmap='viridis',levels=np.arange(1,13,1))

plot_pacific_subplot(co2_cafe.sel(ensemble=25).groupby('time.month').mean().idxmax(dim='month'),sb=323,title='Peak Month  CO2 Cafe',cmap='viridis',levels=np.arange(1,13,1))#,levels=np.arange(-12,12,1))
plot_pacific_subplot(co2_land.groupby('time.month').mean().idxmax(dim='month'),sb=324,title='Peak Amplitude CO2 Landschutzer',cmap='viridis',levels=np.arange(1,13,1))#,levels=np.arange(-12,12,1))
#plot_pacific_subplot(co2_rodenbeck.mean(dim='time')*-1)
plot_pacific_subplot(npp_cafe_25.groupby('time.month').mean().idxmax(dim='month'),sb=325,title='Peak Month NPP Cafe 25',cmap='viridis',levels=np.arange(1,13,1))
#plot_pacific_subplot(npp_cafe_23.mean(dim='time'),sb=326,title='NPP Cafe 25')
plot_pacific_subplot(np_dat.groupby('time.month').mean().idxmax(dim='month'),sb=326,title='Peak Month NPP Obs',cmap='viridis',levels=np.arange(1,13,1))
plt.tight_layout()
plt.show()

# + tags=[]
#plot_pacific(sst_rey.mean(dim='time'))
#plot_pacific(sst_cafe.sel(ensemble=25).groupby('time.month').mean().std(dim='month'))
#plot_pacific(sst_rey.groupby('time.month').mean().std(dim='month'))
#plot_pacific(co2_cafe.sel(ensemble=25).groupby('time.month').mean().std(dim='month'))
#plot_pacific(co2_land.groupby('time.month').mean().std(dim='month'))
#plot_pacific((co2_rodenbeck*-1).groupby('time.month').mean().std(dim='month'))
#plot_pacific(npp_cafe_25.groupby('time.month').mean().std(dim='month'))
#plot_pacific(npp_cafe_23.groupby('time.month').mean().std(dim='month'))
#plot_pacific(np_dat.groupby('time.month').mean().std(dim='month'))
# -

# # Figure 4: Detrended, Deseasonalised Interannual variability



# +
plt.figure(figsize=(20,12))
plt.subplot(311)
deseasonaliser(xarray_detrend(sst_cafe.sel(ensemble=25),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
deseasonaliser(xarray_detrend(sst_cafe.sel(ensemble=25),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2,linestyle=':')
deseasonaliser(xarray_detrend(sst_rey,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
plt.legend(['CAFE ens 25','CAFE ens 23','Reynolds OISST'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('SST Interannual Variability')

plt.subplot(312)
deseasonaliser(xarray_detrend(co2_cafe.sel(ensemble=25),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
deseasonaliser(xarray_detrend(co2_cafe.sel(ensemble=23),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2,linestyle=':')
deseasonaliser(xarray_detrend(co2_cafe_natural.sel(ensemble=25),keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='purple')
deseasonaliser(xarray_detrend(co2_land,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
deseasonaliser(xarray_detrend(co2_rodenbeck*-1,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='g',linewidth=2,linestyle=':')
plt.legend(['Cafe ens 25','Cafe ens 23','Cafe Natural CO2','landchutzer','rodenbeck'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.ylim([-1.5,1.5])
plt.title('CO2 Interannual Variability')

plt.subplot(313)
deseasonaliser(xarray_detrend(npp_cafe_25,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
deseasonaliser(xarray_detrend(npp_cafe_25_sed,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linestyle=':',linewidth=2)
deseasonaliser(xarray_detrend(npp_cafe_23,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='g',linewidth=2)
deseasonaliser(xarray_detrend(np_dat,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
plt.legend(['CAFE ens 25','cafe 25 Sediment', 'CAFE ens 23','laws x cafe obs'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('NPP Interannual Variability')
plt.tight_layout()

# +
plt.figure(figsize=(20,12))
plt.subplot(311)
deseasonaliser(xarray_detrend(sst_cafe.sel(ensemble=25),keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
deseasonaliser(xarray_detrend(sst_cafe.sel(ensemble=25),keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='k',linewidth=2,linestyle=':')
deseasonaliser(xarray_detrend(sst_rey,keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
plt.legend(['CAFE ens 25','CAFE ens 23','Reynolds OISST'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('SST Interannual Variability')

plt.subplot(312)
deseasonaliser(xarray_detrend(co2_cafe.sel(ensemble=25),keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
deseasonaliser(xarray_detrend(co2_cafe.sel(ensemble=23),keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='k',linewidth=2,linestyle=':')
deseasonaliser(xarray_detrend(co2_cafe_natural.sel(ensemble=25),keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='purple')
deseasonaliser(xarray_detrend(co2_land,keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
deseasonaliser(xarray_detrend(co2_rodenbeck*-1,keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='g',linewidth=2,linestyle=':')
plt.legend(['Cafe ens 25','Cafe ens 23','Cafe Natural CO2','landchutzer','rodenbeck'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('CO2 Interannual Variability')

plt.subplot(313)
deseasonaliser(xarray_detrend(npp_cafe_25,keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
deseasonaliser(xarray_detrend(npp_cafe_25_sed,keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='k',linestyle=':',linewidth=2)
deseasonaliser(xarray_detrend(npp_cafe_23,keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='g',linewidth=2)
deseasonaliser(xarray_detrend(np_dat,keep_intercept_values=True)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
plt.legend(['CAFE ens 25','cafe 25 Sediment', 'CAFE ens 23','laws x cafe obs'])
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])
plt.title('NPP Interannual Variability')
plt.tight_layout()
# -



plt.figure(figsize=(12,5))
deseasonaliser(xarray_detrend(npp_cafe_25,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='k',linewidth=2)
deseasonaliser(xarray_detrend(npp_cafe_23,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='g',linewidth=2)
deseasonaliser(xarray_detrend(np_dat,keep_intercept_values=False)).mean(dim=['lat','lon']).plot(c='b',linewidth=2,linestyle=':')
plt.xlim([np.datetime64('2000-01-01'),np.datetime64('2020-01-01')])



plt.figure(figsize=(12,5))
deseasonaliser(xarray_detrend(npp_cafe_25,keep_intercept_values=False)).mean(dim=['lat','lon']).plot()
deseasonaliser(xarray_detrend(npp_cafe_23,keep_intercept_values=False)).mean(dim=['lat','lon']).plot()
deseasonaliser(xarray_detrend(np_dat,keep_intercept_values=False)).mean(dim=['lat','lon']).plot()
#deseasonaliser(xarray_detrend(sst_rey,keep_intercept_values=False)).mean(dim=['lat','lon']).plot()

[-15,15,150,275]

# Convert Data for plotting
sst_cafe=bgcdatvs_allens.sst.sel(ensemble=26).sel(time=slice('2000-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('1990-01-01','2020-01-01'))
co2_cafe=bgcdatvs_allens.stf10.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
co2_land=land_co2.sel(time=slice('1990-01-01','2020-01-01'))

# Good test case
plot_pacific_boxes(var_a=sst_cafe,
                   var_b=None,
                   var_a_name='CAFE SST',
                   var_b_name=None,
                   units='SST Deg C',
                   title=' SST SENSE CHECK:',sdate='2000-01-01',
                   detrend=True,deseasonalise=True,plot_decomposition=True,keep_intercept_values=True)

# Good test case
plot_pacific_boxes(var_a=sst_cafe,
                   var_b=None,
                   var_a_name='CAFE SST',
                   var_b_name=None,
                   units='SST Deg C',
                   title=' SST SENSE CHECK:',sdate='2000-01-01',
                   detrend=True,deseasonalise=True,plot_decomposition=True,keep_intercept_values=False,rolling=False)

# Good test case
plot_pacific_boxes(var_a=sst_cafe,
                   var_b=None,
                   var_a_name='CAFE SST',
                   var_b_name=None,
                   units='SST Deg C',
                   title=' SST SENSE CHECK:',sdate='2000-01-01',
                   detrend=True,deseasonalise=True,plot_decomposition=True,keep_intercept_values=False,rolling=True)

rodenbeck_co2

rodenbeck1_CO2=xr.open_dataset('../../rxm599/obs/oc_v2021_daily.nc')

# +
#rodenbeck1_CO2
# -

rodenbeck_co2.mean(dim='time').plot()

plot_pacific_boxes(var_a=bgcdatvs_allens.stf10.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01')),
                   var_b=co2_land,
                   var_c=bgcdatvs_allens.stf07.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01')),
                   var_d=(rodenbeck_co2*-1000)/12,
                   var_a_name='CAFE CO2',
                   var_b_name='LAND CO2',
                   var_c_name='natural co2',
                   var_d_name='rodenbeck co2',
                   units='CO2 flux mol/m2/day',
                   title='Cafe vs Land CO2:',detrend=True,deseasonalise=True,keep_intercept_values=False)

co2_cafe=bgcdatvs_allens.stf10.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
co2_land=land_co2.sel(time=slice('1990-01-01','2020-01-01'))

plot_pacific_boxes(var_a=co2_cafe,
                   var_b=co2_land,
                   var_a_name='CAFE CO2',
                   var_b_name='LAND CO2',
                   units='CO2 flux mol/m2/day',
                   title='Cafe vs Land CO2:',keep_intercept_values=False)

plot_pacific_boxes(var_a=co2_cafe,
                   var_b=co2_land,
                   var_a_name='CAFE CO2',
                   var_b_name='LAND CO2',
                   units='CO2 flux mol/m2/day',
                   title='Cafe vs Land CO2:',detrend=True)

plot_pacific_boxes(var_a=co2_cafe,
                   var_b=co2_land,
                   var_a_name='CAFE CO2',
                   var_b_name='LAND CO2',
                   units='CO2 flux mol/m2/day',
                   title='Cafe vs Land CO2:',detrend=True,deseasonalise=True,plot_decomposition=True)

plot_pacific_boxes(var_a=co2_cafe,
                   var_b=co2_land,
                   var_a_name='CAFE CO2',
                   var_b_name='LAND CO2',
                   units='CO2 flux mol/m2/day',
                   title='Cafe vs Land CO2:',
                   detrend=True,
                   deseasonalise=True)

# +
sst_cafe=bgcdatvs_allens.sst.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('1990-01-01','2020-01-01'))

plot_pacific_boxes(var_a=bgcdatvs_allens.sst.sel(ensemble=25).sel(time=slice('1990-01-01','2020-01-01')),
                   var_b=sst_rey,
                   var_c=bgcdatvs_allens.sst.sel(ensemble=23).sel(time=slice('1990-01-01','2020-01-01')),
                   var_a_name='CAFE SST 25',
                   var_b_name='Reynolds SST',
                   var_c_name='CAFE SST 23',
                   units='SST Deg C',
                   title='Cafe vs Land SST:',
                   detrend=True,deseasonalise=True,keep_intercept_values=True)

# +
sst_cafe=bgcdatvs_allens.sst.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('1990-01-01','2020-01-01'))

plot_pacific_boxes(var_a=bgcdatvs_allens.sst.sel(ensemble=25).sel(time=slice('1990-01-01','2020-01-01')),
                   var_b=sst_rey,
                   var_c=bgcdatvs_allens.sst.sel(ensemble=23).sel(time=slice('1990-01-01','2020-01-01')),
                   var_a_name='CAFE SST 25',
                   var_b_name='Reynolds SST',
                   var_c_name='CAFE SST 23',
                   units='SST Deg C',
                   title='Cafe vs Land SST:',
                   detrend=True,deseasonalise=True,keep_intercept_values=False)

# + tags=[]
sst_cafe=bgcdatvs_allens.sst.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('1990-01-01','2020-01-01'))

plot_pacific_boxes(var_a=bgcdatvs_allens.sst.sel(ensemble=25).sel(time=slice('1990-01-01','2020-01-01')),
                   var_b=sst_rey,
                   var_c=bgcdatvs_allens.sst.sel(ensemble=23).sel(time=slice('1990-01-01','2020-01-01')),
                   var_a_name='CAFE SST 25',
                   var_b_name='Reynolds SST',
                   var_c_name='CAFE SST 23',
                   units='SST Deg C',
                   title='Cafe vs Land SST:',
                   detrend=True,deseasonalise=False,keep_intercept_values=False)

# + tags=[]
sst_cafe=bgcdatvs_allens.sst.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('1990-01-01','2020-01-01'))

plot_pacific_boxes(var_a=bgcdatvs_allens.stf10.sel(ensemble=25).sel(time=slice('1990-01-01','2020-01-01')),
                   var_b=co2_land,
                   var_c=bgcdatvs_allens.stf10.sel(ensemble=23).sel(time=slice('1990-01-01','2020-01-01')),
                   var_a_name='CAFE CO2 25',
                   var_b_name='Land CO2',
                   var_c_name='CAFE CO2 23',
                   units='CO2 flux mmol/day',
                   title='Cafe vs Land CO2 FLUX:',
                   detrend=True,deseasonalise=True,keep_intercept_values=True)

# + tags=[]
sst_cafe=bgcdatvs_allens.sst.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('1990-01-01','2020-01-01'))

plot_pacific_boxes(var_a=bgcdatvs_allens.stf10.sel(ensemble=25).sel(time=slice('1990-01-01','2020-01-01')),
                   var_b=co2_land,
                   var_c=bgcdatvs_allens.stf10.sel(ensemble=23).sel(time=slice('1990-01-01','2020-01-01')),
                   var_a_name='CAFE CO2 25',
                   var_b_name='Land CO2',
                   var_c_name='CAFE CO2 23',
                   units='CO2 flux mmol/day',
                   title='Cafe vs Land CO2 FLUX:',
                   detrend=True,deseasonalise=True,keep_intercept_values=False)

# + tags=[]
sst_cafe=bgcdatvs_allens.sst.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('1990-01-01','2020-01-01'))

plot_pacific_boxes(var_a=bgcdatvs_allens.sst.sel(ensemble=25).sel(time=slice('1990-01-01','2020-01-01')),
                   var_b=sst_rey,
                   var_c=bgcdatvs_allens.sst.sel(ensemble=23).sel(time=slice('1990-01-01','2020-01-01')),
                   var_a_name='CAFE SST 25',
                   var_b_name='Reynolds SST',
                   var_c_name='CAFE SST 23',
                   units='SST Deg C',
                   title='Cafe vs Land SST:',
                   detrend=True,deseasonalise=True)

# + tags=[]
sst_cafe=bgcdatvs_allens.sst.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('1990-01-01','2020-01-01'))

plot_pacific_boxes(var_a=sst_cafe,
                   var_b=sst_rey,
                   var_a_name='CAFE SST',
                   var_b_name='Reynolds SST',
                   units='SST Deg C',
                   title='Cafe vs Land SST:',
                   detrend=False)#,deseasonalise=True)
# -

np1_cafe_25=bgcdatvs_allens.det_export.sel(ensemble=25).sel(time=slice('1990-01-01','2020-01-01'))*6.625
np1_cafe_23=bgcdatvs_allens.det_export.sel(ensemble=23).sel(time=slice('1990-01-01','2020-01-01'))*6.625
np2_cafe=bgcdatvs_allens.trim_export_2d.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))*6.625
np_dat=(np_obs.sel(time=slice('1990-01-01','2020-01-01')))

# +

plot_pacific_boxes(var_a=np1_cafe_25,
                   var_b=np2_cafe_23,
                   var_c=np_dat,
                   var_a_name='CAFE Det Export',
                   var_b_name='CAFE Laws Export',
                   var_c_name='Laws Cafe export',
                   units='New Production / Export mmol / m2/ day',
                   title='Cafe vs Nic Export:',
                   sdate='2000-01-01')#,detrend=True,deseasonalise=True)
# -
np1_cafe=bgcdatvs_allens.nics_export.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))*6.625
np2_cafe=bgcdatvs_allens.trim_export_2d.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))*6.625
np_dat=(np_obs.sel(time=slice('1990-01-01','2020-01-01')))




# +
#land_co2
#rey_sst

#sst #regrid?
#npp CAFE
# *fr.laws2
# +
# # Seasonality?
# -

# Convert Data for plotting
sst_cafe=bgcdatvs_allens.sst.sel(time=slice('2000-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('2000-01-01','2020-01-01'))
co2_cafe=bgcdatvs_allens.stf10.sel(time=slice('2000-01-01','2020-01-01'))
co2_land=land_co2.sel(time=slice('2000-01-01','2020-01-01'))

# +
#sst_cafe
#sst_rey
# -



# +
# Convert Data for plotting
sst_cafe=bgcdatvs_allens.sst.sel(time=slice('2000-01-01','2020-01-01'))
sst_rey=rey_sst.sel(time=slice('2000-01-01','2020-01-01'))
co2_cafe=bgcdatvs_allens.stf10.sel(time=slice('2000-01-01','2020-01-01'))
co2_land=land_co2.sel(time=slice('2000-01-01','2020-01-01'))

npp_cafe_25=bgcdatvs_allens.pprod_gross_2d.sel(ensemble=25).sel(time=slice('1990-01-01','2020-01-01'))*6.625
npp_cafe_23=bgcdatvs_allens.pprod_gross_2d.sel(ensemble=23).sel(time=slice('1990-01-01','2020-01-01'))*6.625
np2_cafe=bgcdatvs_allens.trim_export_2d.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))*6.625
np_dat=(np_obs.sel(time=slice('1990-01-01','2020-01-01')))
plot_pacific_boxes(var_a=npp_cafe_25,
                   var_b=npp_cafe_23,
                   var_c=npp.sel(time=slice('1990-01-01','2020-01-01'))/12,
                   var_a_name='CAFE GPP 25',
                   var_b_name='CAFE GPP 23',
                   var_c_name='Cafe NPP',
                   units='New Production / Export mmol / m2/ day',
                   title='Cafe vs Nic Export:',
                   sdate='2000-01-01',detrend=True,deseasonalise=True,keep_intercept_values=False)
# -

np2_cafe=bgcdatvs_allens.trim_export_2d.sel(ensemble=26).sel(time=slice('1990-01-01','2020-01-01'))*6.625
np_dat=(np_obs.sel(time=slice('1990-01-01','2020-01-01')))



sst_rey.groupby('time.month').mean().mean(dim=['lat','lon']).plot()

plt.figure(figsize=(15,8))
for i in np.arange(15,30):
    sst_cafe.groupby('time.month').mean().mean(dim=['lat','lon']).sel(ensemble=i+1).plot()
sst_rey.groupby('time.month').mean().mean(dim=['lat','lon']).plot(c='k')

plt.figure(figsize=(15,8))
for i in np.arange(15,30):
    co2_cafe.groupby('time.month').mean().mean(dim=['lat','lon']).sel(ensemble=i+1).plot()
co2_land.groupby('time.month').mean().mean(dim=['lat','lon']).plot(c='k')

sst_rey.groupby('time.month').mean().mean(dim=['month']).plot()

sst_rey.groupby('time.month').mean().std(dim=['month']).plot() #STd of Months




# +


def plot_pacific(dat):

    fig = plt.figure(figsize=(12,7))

    # this declares a recentered projection for Pacific areas
    proj = ccrs.PlateCarree(central_longitude=180)
    proj._threshold /= 20.  # to make greatcircle smooth

    ax = plt.axes(projection=proj)
    # set appropriate extents: (lon_min, lon_max, lat_min, lat_max)
    ax.set_extent([120, 290, -20, 20], crs=ccrs.PlateCarree())

    geodetic = ccrs.Geodetic()
    plate_carree = ccrs.PlateCarree(central_longitude=180)

    lonm,latm=np.meshgrid(sst_rey.lon,sst_rey.lat)
    g=ax.contourf(dat.lon,dat.lat,dat, transform=ccrs.PlateCarree(),cmap='bwr')
    plt.colorbar(g,ax=ax,location='bottom',orientation='horizontal')
    # plot greatcircle arc

    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100, edgecolor='k')
    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.coastlines()

    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linewidth=0.3)

    plt.show()
    
    
# #Seasonality?
#plot_pacific(sst_rey.groupby('time.month').mean().std(dim=['month']))

stat=xarray_get_trend(sst_rey)
plot_pacific(stat.sel(parameter=0)*365)

stat=xarray_get_trend(sst_cafe)
plot_pacific(stat.sel(parameter=0)*365)
# -









# ?plt.colorbar

plt.figure(figsize=(15,8))
#for i in np.arange(15,30):
#    xarray_detrend(co2_cafe).groupby('time.month').mean().mean(dim=['lat','lon']).sel(ensemble=i+1).plot()
#xarray_detrend(co2_land).groupby('time.month').mean().mean(dim=['lat','lon']).plot(c='k')





















