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
    landschutzer_CO2=(landschutzer_CO2*12)/365 #to grams/day
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


def process_co2_land_trends(force=False,
                            processed_fp='/g/data/xv83/np1383/processed_data/'):
    
    paths=['global','eqpac']
    for path in paths:
        filepath=processed_fp+'/obs/landshutzer_'+path+'_regrid.nc'
         
        if check_existing_file(filepath)==True:

            land_obs=xr.open_dataset(filepath)
            land_obs_tr_1982=calc_longterm_trends(land_obs.fgco2_smoothed,'1982')
            land_obs_tr_2000=calc_longterm_trends(land_obs.fgco2_smoothed,'2000')
            
            if check_existing_file(processed_fp+'obs/landshutzer_'+path+'_regrid_trend_1982.nc',force)==False:
                print('Saving 1982 CO2 flux trends')
                land_obs_tr_1982.to_netcdf(processed_fp+'obs/landshutzer_'+path+'_regrid_trend_1982.nc')
                
            
            if check_existing_file(processed_fp+'obs/landshutzer_'+path+'_regrid_trend_2000.nc',force)==False:
                print('Saving 2000 CO2 flux trends')
                land_obs_tr_2000.to_netcdf(processed_fp+'obs/landshutzer_'+path+'_regrid_trend_2000.nc')
        
        
        
        
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

    
def make_earth_grid_m2(processed_path='/g/data/xv83/np1383/processed_data/'):
    boxlo,boxla=np.array(np.meshgrid(np.arange(-179.5,179.5,1),np.arange(-89.5,89.5,1)))
    actual_grid=np.cos(np.radians(abs(boxla)))*(111.1*111.1*1000*1000)
    grid_nc=xr.DataArray(actual_grid,coords={'lat':boxla[:,1],'lon':boxlo[1,:]},dims=['lat','lon'])
    lat_size=110567 #in m
    grid_nc['m2']=grid_nc#*lat_size
    grid_nc=grid_nc['m2']
    grid_nc.to_netcdf(processed_path+'earth_m2.nc',engine='h5netcdf',mode='w')
    return True


def earth_grid_m2_rodenbeck(rodenbeck_path,processed_path='/g/data/xv83/np1383/processed_data/'):
    # Uses regridded CO2
    # If using ,='../../rxm599/obs/oc_v2021_daily.nc' need to change the 1 below to 2.5 and 2. 
#178.8 -176.2 ... 176.2 178.8
# (-178.75,178.75+2.5,2.5)
    boxlo,boxla=np.array(np.meshgrid(np.arange(0,360,1),np.arange(-89,89+1,1)))
    
    actual_grid=np.cos(np.radians(abs(boxla)))*(111.1*111.1*1000*1000)
    grid_nc=xr.DataArray(actual_grid,coords={'lat':boxla[:,1],'lon':boxlo[1,:]},dims=['lat','lon'])
    lat_size=110567 #in m
    grid_nc['m2']=grid_nc#*lat_size
    grid_nc=grid_nc['m2']
    
    rodenbeck=xr.open_dataset(rodenbeck_path)
    regridder = xe.Regridder(grid_nc, rodenbeck, 'bilinear',reuse_weights=False)
    grid_nc_regrid=regridder(grid_nc)
    
    if check_existing_file(processed_path+'earth_m2_regrid_rodenbeck.nc',force=True)==False:
        grid_nc_regrid.to_netcdf(processed_path+'earth_m2_regrid_rodenbeck.nc',engine='h5netcdf')
    return True


    
def proc_rodenbeck(cuttropics=False,force=False,verbose=True,
                      obs_fp='../../rxm599/obs/',
                      processed_path='/g/data/xv83/np1383/processed_data/'):
    #Load and process landschutzer data
    rodenbeck_CO2=xr.open_dataset(obs_fp+'oc_v2021_daily.nc',chunks={'time':1}).co2flux_ocean
    
    #Rename time and convert to monthly and first day of month
    rodenbeck_CO2=rodenbeck_CO2.rename({'mtime':'time'})
    
    rodenbeck_CO2=rodenbeck_CO2.resample(time='M').mean()
    rodenbeck_CO2['time']=rodenbeck_CO2['time'].astype('datetime64[M]')  
    rodenbeck_CO2= rodenbeck_CO2.assign_coords(lon=(rodenbeck_CO2.lon % 360)).roll(lon=(rodenbeck_CO2.sizes['lon']),roll_coords=False).sortby('lon')		#EPIC 1 line fix for the dateline problem.

    rodenbeck_CO2_units=(rodenbeck_CO2*(10**15))/365 #Convert from PgC/Yr/Grid to gC/day/Grid
    #Regrid only the new version not the climatology 
    rodenbeck_CO2_units=rodenbeck_CO2_units #from g CO2 to C.W
    
    #if verbose: print(rodenbeck_CO2)
   
    # Xarray craziness means we cant do all of these calculations at once. Calculate intermediate step (Delete file after?)
    intermediate_path=processed_path+'obs/rodenbeck_intermediate_regrid.nc'
    if check_existing_file(intermediate_path,False)==False:
        rodenbeck_CO2_units.to_netcdf(intermediate_path)
    rodenbeck_CO2.close()
    rodenbeck_CO2_units.close()
    
    if verbose: print('Saved and reloaded intermediate step, now regridding')
    
    rodenbeck_CO2_intermediate=xr.open_dataset(intermediate_path)
    if verbose: print('Normalising to m2')
    earth_grid_m2_rodenbeck(rodenbeck_path=intermediate_path)
    grid_nc=xr.open_dataset(processed_path+'earth_m2_regrid_rodenbeck.nc',engine='h5netcdf')
    rodenbeck_CO2_intermediate_m2=(rodenbeck_CO2_intermediate/(grid_nc.m2*5))#.m2.plot()  *5 is for the 2.5 to 2 ratio. Regrid weirdness easiest fix?
    
    
    cafe=xr.open_dataset(processed_path+'cafe/global/stf10_ensmean_1982.nc')
    regridder = xe.Regridder(rodenbeck_CO2_intermediate_m2, cafe, 'bilinear',reuse_weights=False)
    rodenbeck_CO2_regrid=regridder(rodenbeck_CO2_intermediate_m2)
    if verbose: print('Regridded')
    #rodenbeck_CO2=rodenbeck_CO2.chunk(chunks)
    


    savepath='global'
    if cuttropics==True:
        savepath='eqpac'
        rodenbeck_CO2_regrid=rodenbeck_CO2_regrid.sel(lat=slice(-20,20),lon=slice(120,290))

        
    filepath=processed_path+'obs/rodenbeck_'+savepath+'_regrid.nc'
    print(rodenbeck_CO2_regrid.nbytes/1e9)
    rodenbeck_CO2_regrid.load()
    print('loaded')
    print(rodenbeck_CO2_regrid)
    if check_existing_file(filepath,force)==False:
        print('saving to: '+filepath)
        rodenbeck_CO2_regrid.to_netcdf(filepath)
    else:
        print('Not resaving Rodenbeck: '+filepath)


def process_co2_rodenbeck_trends(force=False,
                            processed_fp='/g/data/xv83/np1383/processed_data/'):
    
    paths=['global','eqpac']
    for path in paths:
        filepath=processed_fp+'/obs/rodenbeck_'+path+'_regrid.nc'
         
        if check_existing_file(filepath)==True:

            rodenbeck_obs=xr.open_dataset(filepath).co2flux_ocean
            rodenbeck_obs_tr_1982=calc_longterm_trends(rodenbeck_obs,'1982')
            rodenbeck_obs_tr_2000=calc_longterm_trends(rodenbeck_obs,'2000')
            
            if check_existing_file(processed_fp+'obs/rodenbeck_'+path+'_regrid_trend_1982.nc',force)==False:
                print('Saving 1982 CO2 flux trends')
                rodenbeck_obs_tr_1982.to_netcdf(processed_fp+'obs/rodenbeck_'+path+'_regrid_trend_1982.nc')
                
            
            if check_existing_file(processed_fp+'obs/rodenbeck_'+path+'_regrid_trend_2000.nc',force)==False:
                print('Saving 2000 CO2 flux trends')
                rodenbeck_obs_tr_2000.to_netcdf(processed_fp+'obs/rodenbeck_'+path+'_regrid_trend_2000.nc')
        

                
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
    #print(hm)
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


def convert_trim_fratios(trim_path='../external_data/fratios/SIMPLE_TRIM_output.nc',
                         regrid_path='../processed_data/rey_eqpac_sst_rg.nc'):
        '''
        This data is sourced from Tim DeVris, 2017
        https://tdevries.eri.ucsb.edu/models-and-data-products/
        DeVries, T., and Weber, T. (2017). The export and fate of organic matter in the ocean: New constraints from combining satellite and oceanographic tracer observations: EXPORT AND FATE OF MARINE ORGANIC MATTER. Global Biogeochem. Cycles 31, 535–555.
    
        This function will return the regridded eqpac TRIM ef ratio.
    
        '''
        trim=xr.open_dataset(trim_path)
        
        #ratio of sinking particle flux at the base of the euphotic zone to the NPP at each grid point)
        # This loops through to see the difference between each
        # Tim said that he just averages the 
        # for i in range(0,12):
        #     ver=trim.sel(version=i)
        #     efratio1=ver.NPP/ver.FPOCex
        #     efratio2=ver.FPOCex/ver.NPP
        #     efratio2.T.plot.contourf(levels=np.arange(0,0.425,0.025),cmap='viridis')
        #     plt.suptitle(str(i))
        #     plt.show()
        #     print(i)
        #     print((((ver.FPOCex/1000)*12)*ver.Area).sum().values/1e15) #Global integrated carbon removal
        # #trim=trim.set_coords('version')
        
        ratio=(trim.FPOCex/trim.NPP).mean(dim='version')
        ratio.name='avg'
        std=(trim.FPOCex/trim.NPP).std(dim='version')
        std.name='stdev'
        ratio=xr.merge([ratio,std])
        ourset=trim.mean(dim='version')
        print((((ourset.FPOCex/1000)*12)*ourset.Area).sum().values/1e15) #Global integrated carbon removal
        #sel(version=5)
        ratio['latitude']= trim.LAT.values[0][0]
        ratio['longitude']= trim.LON.values[0][:,0]
        
        ratio=ratio.rename({'latitude':'lat','longitude':'lon'})
        ratio=ratio.where((ratio>0)&(ratio<100))
        
        eqpac_trim=ratio.sel(lat=slice(-21,21),lon=slice(119,301))
        eqpac_trim['avg']=eqpac_trim.avg.T
        eqpac_trim['stdev']=eqpac_trim.stdev.T
        
        #eqpac_trim.stdev.plot.contourf(levels=np.arange(0,0.425,0.025))
         
        to_regridder=xr.open_dataset(regrid_path)

        regridder = xe.Regridder(eqpac_trim, to_regridder, 'bilinear')
        eqpac_trim_grid=regridder(eqpac_trim)
        return eqpac_trim_grid

if __name__ == '__main__':
    pass