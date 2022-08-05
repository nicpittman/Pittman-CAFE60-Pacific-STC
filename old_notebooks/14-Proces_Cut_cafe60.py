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
from xgcm import Grid
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
    cluster.scale(cores=64)
    #cluster.adapt(minimum=2, maximum=16)
    cluster
    #client

cluster


client


# ls /g/data/xv83/dcfp/CAFE60v1/

# +
# Area grid 
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


# LOAD CAFE
bgc=xr.open_zarr('/g/data/xv83/dcfp/CAFE60v1/ocean_bgc_month.zarr.zip')
bgc=bgc.rename({'xt_ocean':'lon','yt_ocean':'lat'})
bgc['lon']=bgc['lon']+360
eqpac=bgc.sel(lon=slice(120,290),lat=slice(-40,40))
eqpac['time']=eqpac.time.astype('datetime64[M]')
eqpac



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
#sst=xr.open_dataset('../processed_data/obs/sst.mnmean.regrid.eqpac.nc')
#sst=xr.open_dataset('../processed_data/rey_eqpac_sst_rg.nc').__xarray_dataarray_variable__

npp=xr.open_dataset('../processed_data/npp_rg/avg_npp_cafe.nc').avg_npp.chunk('auto')
npp_rg=xr.open_dataset('../processed_data/npp_rg/avg_npp_rg_cafe.nc').avg_npp
#npp_vgpm=xr.open_dataset('../processed_data/npp_rg/avg_npp_rg_vgpm.nc').avg_npp
#npp_cbpm=xr.open_dataset('../processed_data/npp_rg/avg_npp_rg_cbpm.nc').avg_npp
#npp_eppley=xr.open_dataset('../processed_data/npp_rg/avg_npp_rg_eppley.nc').avg_npp
chl=xr.open_dataset('../processed_data/obs/TPCA_month_regrid.nc').__xarray_dataarray_variable__
chl_modis=xr.open_dataset('../processed_data/obs/TPCA_modis_month_regrid.nc').__xarray_dataarray_variable__

tpca=xr.open_dataset('../processed_data/obs/TPCA_month_regrid.nc').__xarray_dataarray_variable__#.mean(dim='time').plot(vmin=0,vmax=0.3)
tpca_sw=xr.open_dataset('../processed_data/obs/TPCA_sw_month_regrid.nc').__xarray_dataarray_variable__
tpca_mod=xr.open_dataset('../processed_data/obs/TPCA_modis_month_regrid.nc').__xarray_dataarray_variable__

fr=xr.open_dataset('../processed_data/fratios_rg.nc')

np_obs=(npp_rg/12)*fr.laws2011a
laws2011a=((0.5857-0.0165*rey_sst)*npp_rg)/(51.7+npp_rg)
# +
#npp.mean(dim='time').plot()
# -


phys_eqpac


client
# +
cut_eqpac_cafe=False
run_dic=False
run_bgc_cut_cafe=False
process_interim_only_wanted_vars=False
run_trends=False
dont_run=False # Bulk ensemble save.
calc_density=False

ensemble = 25
if cut_eqpac_cafe==True:
    eqpac_cut=phys_eqpac.sel(ensemble=ensemble,st_ocean=slice(0,1000),sw_ocean=slice(0,1000))
    eqpac_cut['sw_ocean']=eqpac_cut.sw_ocean*-1
    eqpac_cut['st_ocean']=eqpac_cut.st_ocean*-1
    #'salt','temp','sst','age_global','u','v','wt','tx_trans','ty_trans'
    varz=['tx_trans_gm','ty_trans_gm']
    for var in varz:
        ph=eqpac_cut[var].chunk('auto')
        
        #ph['st_ocean']=zc.st_ocean
        print(ph.nbytes/1e9)
        print(var)
        ph.to_netcdf(f'../processed_data/physics_ds/{var}_physics_feb7.nc')
        print(f'saved {var}')
        ph.close()
        eqpac_cut.close()


# -
run_dic=False
if run_dic==True:
    dic=eqpac.sel(ensemble=25,st_ocean=slice(0,1000))[['dic','adic']].chunk('auto')
    dic['st_ocean']=dic.st_ocean*-1
    print(dic)
    print(dic.nbytes/1e9)
    dic.to_netcdf('../processed_data/physics_ds/dic_pac_feb7.nc')
    anth_dic=dic['adic']-dic['dic']
    anth_dic.chunk('auto').to_netcdf('../processed_data/physics_ds/anth_DIC_pac_feb7.nc')

if run_bgc_cut_cafe==True:
    eqpac_cut=eqpac.sel(ensemble=ensemble,st_ocean=0,method='nearest')
    varz=['pprod_gross_2d','pco2','paco2','stf10','stf07','pprod_gross_2d','NEWPROD']
    for var in varz:
        if var=='NEWPROD':
            pass
            # Do something
        else:
            ph=eqpac_cut[var]

        #ph['st_ocean']=zc.st_ocean
        print(ph.nbytes/1e9)
        print(var)
        ph.to_netcdf(f'../processed_data/physics_ds/{var}_bgc_feb7.nc')

# +
# Calculate Density

if calc_density==True:
#import seawater

#def dens_wrapper(s,t,p):
#    print(s,t,p)
#    return seawater.eos80.dens(s,t,p)

    cafe_salt=xr.open_dataset('../processed_data/physics_ds/salt_physics_feb7.nc')
    cafe_temp=xr.open_dataset('../processed_data/physics_ds/temp_physics_feb7.nc')

    dense_calc=cafe_salt.chunk('auto')
    dense_calc['temp']=cafe_temp.temp.chunk('auto')

    pdensity=xr.apply_ufunc(seawater.eos80.pden,dense_calc.salt,dense_calc.temp,dense_calc.st_ocean,dask='parallelized')#,input_core_dims=['salt','temp','st_ocean'])
    pdensity.to_netcdf('../processed_data/physics_ds/potential_density_physics_feb7.nc')
    
    density=xr.apply_ufunc(seawater.eos80.dens,dense_calc.salt,dense_calc.temp,dense_calc.st_ocean,dask='parallelized')#,input_core_dims=['salt','temp','st_ocean'])
    density.to_netcdf('../processed_data/physics_ds/density_physics_feb7.nc')

# +
#Not running this one right now

if process_interim_only_wanted_vars==True:
    # ALL ensembles
    # Super computer intensive. See commented SLURM code at top. 2 clusters. Maybe add more or more workers. Memory limited.
    # BGC
    print('Loading')
    bgcdat=eqpac#.chunk(dict(time=-1))#,lat=lat,lon=lon
    bgcdatvs=bgcdat[['pprod_gross_2d','stf10','stf07']]*60*60*24#.sel(st_ocean=slice(0,depth_integration)).mean(dim='st_ocean')
    print('Cutting')
    #ex=(bgcdat[['det']].sel(st_ocean=slice(0,depth_integration)).integrate(coord='st_ocean'))*0.1#10m/s)#.rename({'det':'nic_export'})
    
    # Fix 100m 
    # Det at 100m * 10m/day mmol/m3 *m/day mmol/m2/day
    bgcdatvs['det_sediment']=bgcdat.det_sediment*6.625*12
    bgcdatvs['det_export']=(bgcdat[['det']].sel(st_ocean=100,method='nearest')*10*6.625*12).det#10m/day)#.rename({'det':'nic_export'})
    bgcdatvs['pco2']=bgcdat.pco2
    print('Detritus calc')


    #bgcdatvs[['pprod_gross_2d','stf10','stf07']]=bgcdatvs[['pprod_gross_2d','stf10','stf07']]*60*60*24 # From seconds to day..
    #print('to Day')
    #bgcdatvs['trim_export']=bgcdatvs.pprod_gross*fr.trim
    #bgcdatvs['laws_export']=bgcdatvs.pprod_gross*fr.laws2011a.mean(dim='time')
    #bgcdatvs['trim_export_2d']=bgcdatvs.pprod_gross_2d*fr.trim
    #bgcdatvs['laws_export_2d']=bgcdatvs.pprod_gross_2d*fr.laws2011a.mean(dim='time')
    print('Export Calc')
    print(bgcdatvs.nbytes/1e9)
    print(bgcdatvs)
    print('saving')
    bgcdatvs.to_netcdf('../processed_data/model_proc_temp/epac_bgc_all_ensembles.nc')
    print('saved. onto physics')
    # Physics Now
    physdat=phys_eqpac#.chunk(dict(time=-1))#sel(ensemble=ens,method='nearest')
    physdatvs=physdat[['temp','sst']].sel(st_ocean=slice(0,depth_integration)).mean(dim='st_ocean')

    physdatvs.to_netcdf('../processed_data/model_proc_temp/epac_phys_all_ensembles.nc')

# +
#bgcdatvs=xr.open_dataset('../processed_data/model_proc_temp/epac_bgc.nc')
#physdatvs=xr.open_dataset('../processed_data/model_proc_temp/epac_phys.nc')

if dont_run==True:
    bgcdatvs_allens=xr.open_dataset('../processed_data/model_proc_temp/epac_bgc_all_ensembles.nc')
    physdatvs_allens=xr.open_dataset('../processed_data/model_proc_temp/epac_phys_all_ensembles.nc')

    # Combine BGC and physics DFs.
    physdatvs_allens['time']=bgcdatvs_allens['time']
    bgcdatvs_allens['temp']=physdatvs_allens['temp']
    bgcdatvs_allens['sst']=physdatvs_allens['sst']


    # Convert Data for plotting
    sst_cafe=bgcdatvs_allens.sst.sel(time=slice('1998-01-01','2020-01-01'))
    sst_rey=rey_sst.sel(time=slice('1998-01-01','2020-01-01'))
    co2_cafe=bgcdatvs_allens.stf10.sel(time=slice('1998-01-01','2020-01-01'))
    co2_cafe_natural=bgcdatvs_allens.stf07.sel(time=slice('1998-01-01','2020-01-01'))
    co2_land=land_co2.sel(time=slice('1998-01-01','2020-01-01'))
   


    npp_cafe_25=bgcdatvs_allens.det_export.sel(ensemble=25).sel(time=slice('1998-01-01','2020-01-01'))/12#*6.625
    npp_cafe_25_sed=bgcdatvs_allens.det_sediment.sel(ensemble=25).sel(time=slice('1998-01-01','2020-01-01'))/12#*6.625

    npp_cafe_23=bgcdatvs_allens.det_export.sel(ensemble=23).sel(time=slice('1998-01-01','2020-01-01'))/12#*6.625
    np2_cafe=bgcdatvs_allens.trim_export_2d.sel(ensemble=26).sel(time=slice('1998-01-01','2020-01-01'))/12#*6.625
    np_dat=(np_obs.sel(time=slice('1998-01-01','2020-01-01')))
    
# -




# +
# Load the data we created

cafe_u=xr.open_dataset('../processed_data/physics_ds/u_physics_feb7.nc').u
cafe_v=xr.open_dataset('../processed_data/physics_ds/v_physics_feb7.nc').v
cafe_wt=xr.open_dataset('../processed_data/physics_ds/wt_physics_feb7.nc').wt
cafe_temp=xr.open_dataset('../processed_data/physics_ds/temp_physics_feb7.nc').temp
cafe_sst=xr.open_dataset('../processed_data/physics_ds/sst_physics_feb7.nc').sst
cafe_age=xr.open_dataset('../processed_data/physics_ds/age_global_physics_feb7.nc').age_global

dic=xr.open_dataset('../processed_data/dic_pac_feb7.nc') #physics_ds
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
# +
#dic_tx_v1
# -



import shutil


# +
def calculate_rolling_trends(dat,title,force=True,zarr=False,npp=False):

    trend_ens_fp=f'../processed_data/var_ensembles/{title}_trend_ensemble.nc'
    if ((os.path.isfile(trend_ens_fp)==False)|(force==True)):
        print(f'Calculating trend ensemble for {title}')
        holder=[]
        for i in np.arange(0,5*12,1):
            tlen=17

            start_day_iter=np.datetime64('1998-01')+np.timedelta64(i,'M')
            end_day_iter=start_day_iter+np.timedelta64(tlen,'Y')



            iter_test=dat.sel(time=slice(start_day_iter,end_day_iter))


            hh_iter=xarray_get_trend(iter_test)*365#calculate_trend(iter_test)
            hh_iter.name=f'{start_day_iter} to {end_day_iter}'
            if npp==True:
                print(f'saving {hh_iter}')
                hh_iter.to_netcdf(f'../processed_data/var_ensembles/npp_slices/npp_{hh_iter.name}.nc')
            print(hh_iter.name)
            holder.append(hh_iter)
            
        time_period_ensemble=xr.concat(holder,dim='timeperiod')
        if os.path.exists(trend_ens_fp):
            os.remove(trend_ens_fp)
        print('saving now')
        #print(time_period_ensemble.chunks)
        #return time_period_ensemble 
        # .chunk('auto')


        if zarr==False:
            time_period_ensemble.to_netcdf(trend_ens_fp)
        print('saved')
    else:
        print(f'Loading trend ensemble for {title}')
        time_period_ensemble=xr.open_dataset(trend_ens_fp)
    return time_period_ensemble#

#if zarr==True:
#            if i==0:
#                 if os.path.exists(trend_ens_fp[:-2]+'zarr'):
#                    #os.rmdir()
#                    shutil.rmtree(trend_ens_fp[:-2]+'zarr')
#            #for tp in time_period_ensemble.timeperiod:
#            #    print(tp)
#           # print(time_period_ensemble)
#            time_period_ensemble.to_dataset().to_zarr(trend_ens_fp[:-2]+'zarr',mode='a',append_dim='timeperiod')
# -

client

#'rey_sst.sst','cafe_sst', "obs_current.U_320","land_co2","cafe_u_mean_depth","obs_current.V_321","cafe_v_mean_depth",'cafe_co2','cafe_co2_natural'
# This is annoying to run took multiple attempts
# Try saving NPP to ZARR netcdf keeps crashing?
run_trends=True
# 17 YEAR ROLLING TRENDS
npp=npp.chunk(dict(time=-1))
if run_trends==True:
    
    cafe_u_mean_depth=cafe_u.chunk({'st_ocean':-1}).mean(dim='st_ocean')
    cafe_v_mean_depth=cafe_v.chunk({'st_ocean':-1}).mean(dim='st_ocean')
    
    dic_mean_depth=dic_cafe.chunk({'st_ocean':-1}).sel(st_ocean=slice(0,-100)).mean(dim='st_ocean')
    adic_mean_depth=adic_cafe.chunk({'st_ocean':-1}).sel(st_ocean=slice(0,-100)).mean(dim='st_ocean')
    
    trend_vars=['npp']#'rey_sst.sst','cafe_sst', "obs_current.U_320","land_co2","cafe_u_mean_depth","obs_current.V_321","cafe_v_mean_depth",'cafe_co2','cafe_co2_natural','dic_mean_depth','adic_mean_depth','upwelling_cafe','cafe_pprod','npp']
    #trend_vars=['npp']

    for tvar in trend_vars:
        #xr.apply_ufunc(calculate_rolling_trends,eval(tvar),tvar)
        #try:

        tpe=calculate_rolling_trends(eval(tvar),tvar)
        #eval(tvar).to_dataset().apply(calculate_rolling_trends,args={tvar})
        #except:
        #    eval(tvar).apply(calculate_rolling_trends,args={tvar})
        #calculate_rolling_trends(eval(tvar),title=tvar)





dat=xr.open_dataset('../processed_data/var_ensembles/npp_trend_ensemble.nc')
dat['1998-01 to 2015-01'].mean(dim='timeperiod').sel(parameter=0).plot()

# We want the following datasets for the 40NS Pacific
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
#
# Part II
#
# - Land CO2:            
#     - land_co2
#     - co2_rodenbeck
# - Cafe CO2
#     - co2_cafe
# - Cafe Nat Co2
#     - co2_cafe_natural
# - CAFE DIC
# - Land dPCO2
# - cafe dpco2
# - Sat new production
#     - np_dat
# - Cafe new production
#      - npp_cafe_23, np2_cafe
#
#
#
# co2_cafe,sst_rey,sst_cafe
# co2_cafe_natural,co2_land,
#
#

# +
# To calculate dPCO2 we will need to get the ATMOSPHERIC CO2 and calculate the difference. 
# +
# Run tracer transport Calculation
# This will do the two sides of the box but not vertical (bottom / top)> Make new calc to get wt transport?
calculate_horizontal_tracers=False
if calculate_horizontal_tracers==True:
    names=['anth_dic_cafe','dic_cafe']
    for i,tracer in enumerate([anth_dic_cafe,dic_cafe]):
        tracer_u=tracer.rename({'lon':'lon_x'}).chunk('auto')
        tracer_v=tracer.rename({'lat':'lat_x'}).chunk('auto')

        tracer_u_interp=tracer_u.interp(lon_x=cafe_x_tx.lon_x) # U is X (lon)
        tracer_v_interp=tracer_v.interp(lat_x=cafe_y_tx.lat_x) # V is Y (lat)

        # Do need to take mean of adjacent pixels still or not?
        adjacent_boxes=False
        if adjacent_boxes==True:
            tracer_u_tx=tracer_u_interp.rolling(lat=2).mean()*cafe_x_tx.rolling(lon_x=2).mean()
            tracer_v_tx=tracer_v_interp.rolling(lon=2).mean()*cafe_y_tx.rolling(lat_x=2).mean()
        elif adjacent_boxes==False:
            tracer_u_tx=tracer_u_interp*cafe_x_tx
            tracer_v_tx=tracer_v_interp*cafe_y_tx

        # Rename back to main grid so we can use the coords easily
        tracer_u_tx=tracer_u_tx.rename({'lon_x':'lon'})
        tracer_v_tx=tracer_v_tx.rename({'lat_x':'lat'})
        print('saving')
        tracer_u_tx.to_netcdf(f'/g/data/xv83/np1383/processed_data/physics_ds/{names[i]}_u_transport.nc')
        print('u saved')
        tracer_v_tx.to_netcdf(f'/g/data/xv83/np1383/processed_data/physics_ds/{names[i]}_v_transport.nc')
        print('v saved')
        
anth_dic_u_tx=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_cafe_u_transport.nc').__xarray_dataarray_variable__.chunk('auto')*1e6 #to mmol/s
anth_dic_v_tx=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_cafe_v_transport.nc').__xarray_dataarray_variable__.chunk('auto')*1e6 #to mmol/s
dic_u_tx=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_cafe_u_transport.nc').__xarray_dataarray_variable__.chunk('auto')*1e6 #to mmol/s
dic_v_tx=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_cafe_v_transport.nc').__xarray_dataarray_variable__.chunk('auto')*1e6 #to mmol/s
# Units of these should be mmolC/m3 to then SV(kg/s) = mmolC /s so MMol/s *10e6

# Test they look OK

#chunk({'time':-1}).sel(time=slice('2000-01-01','2020-01-01')).
plot=False
if plot==True:
    anth_dic_v_tx.mean(dim='time').sel(lat=-3,method='nearest').plot()#(vmin=-20,vmax=20,cmap='bwr')
    plt.show()
    anth_dic_u_tx.mean(dim='time').sel(lon=180,method='nearest').plot()#(vmin=-20,vmax=20,cmap='bwr')
    plt.show()
    anth_dic_u_tx.mean(dim='time').sel(lat=0,method='nearest').plot()#(vmin=-20,vmax=20,cmap='bwr')

# +
calculate_horizontal_tracers_gm=False
if calculate_horizontal_tracers_gm==True:
    names=['dic_cafe']  #anth_dic_cafe
    for i,tracer in enumerate([dic_cafe]):# anth_dic_cafe
        tracer_u=tracer.rename({'lon':'lon_x'}).chunk('auto')
        tracer_v=tracer.rename({'lat':'lat_x'}).chunk('auto')

        tracer_u_interp=tracer_u.interp(lon_x=cafe_x_tx.lon_x) # U is X (lon)
        tracer_v_interp=tracer_v.interp(lat_x=cafe_y_tx.lat_x) # V is Y (lat)

        # Do need to take mean of adjacent pixels still or not?
        adjacent_boxes=False
        if adjacent_boxes==True:
            tracer_u_tx=tracer_u_interp.rolling(lat=2).mean()*cafe_x_tx.rolling(lon_x=2).mean()
            tracer_v_tx=tracer_v_interp.rolling(lon=2).mean()*cafe_y_tx.rolling(lat_x=2).mean()
        elif adjacent_boxes==False:
            tracer_u_tx=tracer_u_interp*cafe_x_tx_gm
            tracer_v_tx=tracer_v_interp*cafe_y_tx_gm

        # Rename back to main grid so we can use the coords easily
        tracer_u_tx=tracer_u_tx.rename({'lon_x':'lon'})
        tracer_v_tx=tracer_v_tx.rename({'lat_x':'lat'})
        print('saving')
        tracer_u_tx.to_netcdf(f'/g/data/xv83/np1383/processed_data/physics_ds/{names[i]}_u_transport_gm.nc')
        print('u saved')
        tracer_v_tx.to_netcdf(f'/g/data/xv83/np1383/processed_data/physics_ds/{names[i]}_v_transport_gm.nc')
        print('v saved')
        
#anth_dic_u_tx_gm=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_cafe_u_transport_gm.nc').__xarray_dataarray_variable__.chunk('auto')
#anth_dic_v_tx_gm=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_cafe_v_transport_gm.nc').__xarray_dataarray_variable__.chunk('auto')
#dic_u_tx_gm=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_cafe_u_transport_gm.nc').__xarray_dataarray_variable__.chunk('auto')
#dic_v_tx_gm=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_cafe_v_transport_gm.nc').__xarray_dataarray_variable__.chunk('auto')
# Units of these should be mmolC/m3 to then SV(kg/s) = mmolC /s so MMol/s *10e6
# -


xr.open_dataset('../processed_data/physics_ds/anth_DIC_pac_feb7.nc').__xarray_dataarray_variable__


# +
# Load Info for Upwelling


cafe_density=xr.open_dataset('../processed_data/physics_ds/density_physics_feb7.nc').__xarray_dataarray_variable__
cafe_potential_density=xr.open_dataset('../processed_data/physics_ds/potential_density_physics_feb7.nc').__xarray_dataarray_variable__

area_m2=xr.open_dataset('/g/data/xv83/rxm599/area.nc')
area_m2['xt_ocean']=area_m2['xt_ocean']+360
#area_m2['xu_ocean']=area_m2['xu_ocean']+360
area_m2=area_m2.sel(xt_ocean=slice(120,290),yt_ocean=slice(-40,40)).area_t.rename({'xt_ocean':'lon','yt_ocean':'lat'})

dic=xr.open_dataset('../processed_data/physics_ds/dic_pac_feb7.nc')
adic_cafe=dic['adic']
dic_cafe=dic['dic']
anth_dic_cafe=xr.open_dataset('../processed_data/physics_ds/anth_DIC_pac_feb7.nc').__xarray_dataarray_variable__
cafe_wt=xr.open_dataset('../processed_data/physics_ds/wt_physics_feb7.nc').wt
sw_ocean=cafe_wt['sw_ocean'].copy()
cafe_wt=cafe_wt.chunk('auto').rename({'sw_ocean':'st_ocean'}).interp(st_ocean=dic_cafe.chunk('auto').st_ocean)#.chunk('auto')

     
anth_dic_u_tx=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_cafe_u_transport.nc').__xarray_dataarray_variable__.chunk('auto')
anth_dic_v_tx=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_cafe_v_transport.nc').__xarray_dataarray_variable__.chunk('auto')
dic_u_tx=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_cafe_u_transport.nc').__xarray_dataarray_variable__.chunk('auto')
dic_v_tx=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_cafe_v_transport.nc').__xarray_dataarray_variable__.chunk('auto')

upwelling=(cafe_wt*dic_cafe.chunk('auto')*area_m2)#/1e6 # To Sverdrups (From m3?)
anthupwelling=(cafe_wt*anth_dic_cafe.chunk('auto')*area_m2)#/1e6 # To Sverdrups
# -







upwell_fix


upwell_fix


# Density Conversion
density_conversion=False:
if density_conversion==True:
    tracers=['dic_v_tx','anth_dic_v_tx',]
    #'upwelling','anthupwelling','dic_u_tx','anth_dic_u_tx',
    #tracer_name='dic_u_tx'
    for tracer_name in tracers:
        tracer_interpolate=eval(tracer_name).chunk('auto')#eval(tracer_name).chunk('auto')

        if tracer_name[4]=='u':
            tracer_interpolate=tracer_interpolate.interp(lon=cafe_potential_density.lon)
        elif tracer_name[4]=='v':
            tracer_interpolate=tracer_interpolate.interp(lat=cafe_potential_density.lat)

        tracer_interpolate_fix=tracer_interpolate.to_dataset(name=tracer_name).assign_coords({'sw_ocean':sw_ocean})

        tracer_interpolate_fix['density']=cafe_potential_density.chunk('auto')

        density_grid=Grid(tracer_interpolate_fix, coords={'density_grid': {'center':'sw_ocean','outer':'st_ocean'}}, periodic=False)
        #tracer_interpolate_fix['dens_outer'] = density_grid.interp(tracer_interpolate_fix.density, 'density_grid', boundary='extend')
        # Regrid Tracer to the center of the vertical Box.
        tracer_interpolate_fix['tracer'] = density_grid.interp(tracer_interpolate_fix[tracer_name], 'density_grid', boundary='fill')


        theta_target = np.arange(1018, 1032., 0.1)
        tracer_density = density_grid.transform(tracer_interpolate_fix['tracer'].fillna(0),
                                          'density_grid',
                                          theta_target,
                                          method='conservative',
                                          target_data=tracer_interpolate_fix.density.fillna(0))
        print(f'saving {tracer_name} which is {tracer_density.nbytes/1e9}GB')
        tracer_density.to_netcdf(f'/g/data/xv83/np1383/processed_data/physics_ds/{tracer_name}_density.nc')
    #tracer_density.rename({'dens_outer':'density'})


upwell_fix
upwelling_density
# +
dic_u_tx_dens=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_u_tx_density.nc')
anth_dic_v_tx_dens=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_u_tx_density.nc')
dic_u_tx_dens=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_v_tx_density.nc')
anth_dic_v_tx_dens=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_v_tx_density.nc')
dic_w_tx_dens=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/dic_w_tx_density.nc')
#anth_dic_w_tx_dens=xr.open_dataset('/g/data/xv83/np1383/processed_data/physics_ds/anth_dic_w_tx_density.nc')

dic_w_tx_dens.__xarray_dataarray_variable__.sel(density=25,method='nearest').mean(dim='time').plot()
# -



# +
# Load the data we created

cafe_u=xr.open_dataset('../processed_data/physics_ds/u_physics_feb7.nc').u
cafe_v=xr.open_dataset('../processed_data/physics_ds/v_physics_feb7.nc').v
cafe_wt=xr.open_dataset('../processed_data/physics_ds/wt_physics_feb7.nc').wt
cafe_temp=xr.open_dataset('../processed_data/physics_ds/temp_physics_feb7.nc').temp
cafe_sst=xr.open_dataset('../processed_data/physics_ds/sst_physics_feb7.nc').sst
cafe_age=xr.open_dataset('../processed_data/physics_ds/age_global_physics_feb7.nc').age_global
cafe_salt=xr.open_dataset('../processed_data/physics_ds/salt_physics_feb7.nc').salt
cafe_density=xr.open_dataset('../processed_data/physics_ds/density_physics_feb7.nc').__xarray_dataarray_variable__
cafe_potential_density=xr.open_dataset('../processed_data/physics_ds/potential_density_physics_feb7.nc').__xarray_dataarray_variable__
dic=xr.open_dataset('../processed_data/dic_pac_feb7.nc') #physics_ds
upwelling_cafe=cafe_wt.sel(sw_ocean=-100,method='nearest')

adic_cafe=dic['adic']
dic_cafe=dic['dic']
anth_dic_cafe=xr.open_dataset('../processed_data/physics_ds/anth_DIC_pac_feb7.nc').__xarray_dataarray_variable__


cafe_pprod=xr.open_dataset('../processed_data/physics_ds/pprod_gross_2d_bgc_feb7.nc').pprod_gross_2d
cafe_co2=xr.open_dataset('../processed_data/physics_ds/stf10_bgc_feb7.nc').stf10
cafe_co2_natural=xr.open_dataset('../processed_data/physics_ds/stf07_bgc_feb7.nc').stf07
cafe_pCO2=xr.open_dataset('../processed_data/physics_ds/pco2_bgc_feb7.nc').pco2
cafe_paCO2=xr.open_dataset('../processed_data/physics_ds/paco2_bgc_feb7.nc').paco2
#cafe_co2flux=xr.open_dataset('../processed_data/physics_ds/stf10_bgc_feb7.nc').stf10
#cafe_natco2flux=xr.open_dataset('../processed_data/physics_ds/stf07_bgc_feb7.nc').stf07
#age=xr.open_dataset('../processed_data/physics_ds/pCO2_bgc_feb07.nc').age_global


dic_tx_v=xr.open_dataset('../processed_data/dic_v.nc').__xarray_dataarray_variable__
dic_tx_u=xr.open_dataset('../processed_data/dic_u.nc').__xarray_dataarray_variable__
anth_dic_tx_v=xr.open_dataset('../processed_data/dic_v_anth.nc').__xarray_dataarray_variable__
anth_dic_tx_u=xr.open_dataset('../processed_data/dic_u_anth.nc').__xarray_dataarray_variable__

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
# -
cafe_co2_anth=cafe_co2-cafe_co2_natural

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


# +
# Calculate M2 grid 

def make_earth_grid_m2():
    boxlo,boxla=np.array(np.meshgrid(np.arange(0.5,359.5,1),np.arange(-89.5,89.5,1)))
    actual_grid=np.cos(np.radians(abs(boxla)))*(111.1*111.1*1000*1000)
    grid_nc=xr.DataArray(actual_grid,coords={'lat':boxla[:,1],'lon':boxlo[1,:]},dims=['lat','lon'])
    lat_size=110567 #in m
    grid_nc['m2']=grid_nc#*lat_size
    grid_nc1=grid_nc['m2']
    grid_nc1.name='grid_sizes'
    #grid_nc.to_netcdf('processed/earth_m2.nc',engine='h5netcdf',mode='w')
    return grid_nc1


#grid.name='test'
#grid
model_dummy=phys_eqpac.temp
depths=model_dummy.sel(st_ocean=slice(0,500)).st_ocean.values
grid=make_earth_grid_m2().sel(lat=slice(-40,40),lon=slice(120,290)).to_dataset()
#grid.assign_coords(depth=depths.values)#['depths']=depths.values
grid['depth']=depths##.m2#.to_dataset()
l=0
depth_diffs=[]
for i,x in enumerate(grid.depth):
    d=x.values-l
    l=x.values
    #print(d)#x.values)
    depth_diffs.append(d)
grid['depth_diff']=depth_diffs
grid['depth_m3']=grid.m2*grid.depth_diff
grid['depth_m2']=np.sqrt(grid['depth_m3'])
grid['name']='gridsize'
grid
#np.diff(grid.depth.values,axis=0)#()#help(dep)  #.map()#grid.assign_coords(depths=depths)
#gridm2=grid.to_dataset()
#depths  
# -



cafe_density

cafe_density.name='density'

d=cafe_density.to_dataset()
d1=d.set_coords('density')


a=anth_dic_tx_u
a.name='anth_dic_tx'
a=a.to_dataset()
a['density']=d1.density

a1=a.set_coords('density')
a1.chunk('auto')
a1

cafe_density

a1.density.mean(dim=['st_ocean','time']).plot()

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
#
