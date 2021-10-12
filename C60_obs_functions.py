#Make Mega cut function
import xarray as xr
import numpy as np
from dask.distributed import Client
import matplotlib as mpl
from C60_carbon_math import carbon_flux
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import xesmf as xe
from scipy.stats import linregress
import os

from C60_helper_functions import check_existing_file

def proc_landschutzer(cuttropics=False,force=False,
                      obs_fp='/g/data/xv83/np1383/external_data/',
                      processed_path='/g/data/xv83/np1383/processed_data/'):
    #Load and process landschutzer data
    landschutzer_CO2=xr.open_dataset(str(obs_fp)+'co2/landschutzer_co2/spco2_MPI-SOM_FFN_v2020.nc').fgco2_smoothed
    landschutzer_CO2['time']=landschutzer_CO2['time'].astype('datetime64[M]')        
        
    landschutzer_CO2= landschutzer_CO2.assign_coords(lon=(landschutzer_CO2.lon % 360)).roll(lon=(landschutzer_CO2.sizes['lon']),roll_coords=False).sortby('lon')		#EPIC 1 line fix for the dateline problem. #Did dims get broken and moved to sizes?
    #landschutzer_CO2=landschutzer_CO2.sel(lon=slice(120,290),lat=slice(-20,20)).fgco2_smoothed/365 #From per to per day
    landschutzer_CO2=landschutzer_CO2*12 #to grams
    #print(landschutzer_CO2)

    #Regrid only the new version not the climatology 

    cafe=xr.open_dataset(processed_path+'cafe/global/stf10_ensmean_1982.nc')
    regridder = xe.Regridder(landschutzer_CO2, cafe, 'bilinear',reuse_weights=False)
    landschutzer_CO2=regridder(landschutzer_CO2)
    
    savepath='global'
    if cuttropics==True:
        landschutzer_CO2=landschutzer_CO2.sel(lat=slice(-20,20),lon=slice(120,290))
        savepath='eqpac'
        
    filepath=processed_path+'obs/landshutzer'+'_'+savepath+'_regrid.nc'
    if check_existing_file(filepath,force)==False:
        landschutzer_CO2.to_netcdf(filepath)
    else:
        print('Not resaving Landshutzer: '+savepath)

        
        
        
        
def cut_regrid_reynolds_sst(cuttropics=False,
                            force=False,
                            
                            ext_fp='/g/data/xv83/np1383/external_data/',
                            processed_fp='/g/data/xv83/np1383/processed_data/',
                            seamask_flag=False):
    
    sst_cafe=xr.open_dataset(processed_fp+'cafe/global/sst_ensmean_1982.nc').sst
    sst_obs=xr.open_dataset(ext_fp+'sst/sst.mnmean.nc')
    
    #Apply seamask
    if seamask_flag==True:
        raise NotImplementedError('To do.. Really necessary?')
        seamask=xr.open_dataset(ext_fp+'seamask.nc') #Probably doesnt exist.
        seamask= seamask.assign_coords(lon=(seamask.lon % 360)).roll(lon=(seamask.size['lon'] // 2),roll_coords=True)
        seamask=seamask.reindex(lat=seamask.lat[::-1])
        sst_obs_new=sst_obs.sst.where(seamask.seamask==1,np.nan)
    else:
        sst_obs_new=sst_obs
        
    savepath='global'
    if cuttropics==True:
        sst_obs_new=sst_obs_new.sel(lat=slice(20,-20),lon=slice(120,290))
        savepath='eqpac'
        
    sst_obs_new=sst_obs_new.sel(time=slice('1982-01-01','2019-12-01'))
    sst_obs_new=sst_obs_new.reindex(lat=sst_obs.lat[::-1])
    regridder = xe.Regridder(sst_obs_new, sst_cafe, 'bilinear',reuse_weights=False)
    sst_obs_regrid=regridder(sst_obs_new)

    #Could add global if statement here.
    #sst_obs_regrid.to_netcdf('/scratch1/pit071/CAFE60/processed/obs/sst.mnmean.regrid.nc')
    filepath=processed_fp+'obs/sst.mnmean.regrid.'+savepath+'.nc'
    if check_existing_file(filepath,force)==False:
        print('saving: '+filepath)
        sst_obs_regrid.to_netcdf(filepath)
    else:
        print('Not resaving Reynolds SST'+savepath)

        
def cut_process_sst_obs_trends(force=False,
                               processed_fp='/g/data/xv83/np1383/processed_data/'):
    paths=['global','eqpac']
    for path in paths:
        filepath=processed_fp+'obs/sst.mnmean.regrid.'+path+'.nc'
        
        if check_existing_file(filepath)==True:
            sst_obs=xr.open_dataset(filepath)

            sst_obs_tr_1982=calc_longterm_trends(sst_obs.sst,'1982')
            sst_obs_tr_2000=calc_longterm_trends(sst_obs.sst,'2000')

            fp82=processed_fp+'obs/sst.mnmean.regrid.'+str(path)+'.trend.1982.nc'
            if check_existing_file(fp82,force)==False:   
                print('saving 82 sst trends')
                sst_obs_tr_1982.to_netcdf(fp82)
            fp20=processed_fp+'obs/sst.mnmean.regrid.'+str(path)+'.trend.2000.nc'
            if check_existing_file(fp20,force)==False:
                print('saving 2000 sst trends')
                sst_obs_tr_2000.to_netcdf(fp20)

    
def process_co2_land_trends(force=False,
                            processed_fp='/g/data/xv83/np1383/processed_data/'):
    
    paths=['global','eqpac']
    for path in paths:
        filepath=processed_fp+'/obs/landshutzer_'+path+'_regrid.nc'
         
        if check_existing_file(filepath)==True:

            land_obs=xr.open_dataset(filepath)
            land_obs_tr_1982=calc_longterm_trends(land_obs.fgco2_smoothed/365,'1982')
            land_obs_tr_2000=calc_longterm_trends(land_obs.fgco2_smoothed/365,'2000')
            
            if check_existing_file(processed_fp+'obs/landshutzer_'+path+'_regrid_trend_1982.nc',force)==False:
                print('Saving 1982 CO2 flux trends')
                land_obs_tr_1982.to_netcdf(processed_fp+'obs/landshutzer_'+path+'_regrid_trend_1982.nc')
                
            
            if check_existing_file(processed_fp+'obs/landshutzer_'+path+'_regrid_trend_2000.nc',force)==False:
                print('Saving 2000 CO2 flux trends')
                land_obs_tr_2000.to_netcdf(processed_fp+'obs/landshutzer_'+path+'_regrid_trend_2000.nc')


                
def calc_longterm_trends(ds,startday=np.datetime64('1982-01-01'),endday=np.datetime64('2020-01-01')):
    hovmol=ds
    hovmol=hovmol.where(hovmol!=-0.9999,np.nan)
    hm=hovmol.interpolate_na(dim='time').sel(time=slice(startday,endday))
    months=hm.time

    dt_dates=pd.to_numeric(months.values.astype('datetime64[D]'))
    num_dates=dt_dates
    hm['time']=num_dates

    #This will calculate the per pixel trends and pvalues

    time=hm.time.values
    xx=np.concatenate(hm.T)

    tr=[]
    pv=[]
    
    
    for i in range(xx.shape[0]):
        #Add a bit of logic to mask the arrays and if they are empty just fill it with nans.
        x=xx[i,:]
        y=time

        x=np.ravel(x)
        y=np.ravel(y)
        mask=~np.isnan(x)
        x=x[mask]
        y=y[mask]
        if len(x)!=0:
            stat=linregress(y,x)
            tr.append(stat.slope*365)
            pv.append(stat.pvalue)
        else:
            #Fill that row with nans
            stat=linregress(xx[i,:],time)
            tr.append(stat.slope*365)
            pv.append(stat.pvalue)


    tr=np.array(tr).reshape(len(hm.lon),len(hm.lat)).T
    pv=np.array(pv).reshape(len(hm.lon),len(hm.lat)).T

    hh=hm.copy()
    hh=hh.drop('time')
    hh['trend']=(['lat','lon'],tr)
    hh['pval']=(['lat','lon'],pv)
    return hh



if __name__ == '__main__':
    pass