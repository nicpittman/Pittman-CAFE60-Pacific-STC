#Make Mega cut function
import xarray as xr
import numpy as np
from dask.distributed import Client
import matplotlib as mpl
from a_carbon_math import carbon_flux
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import xesmf as xe
from scipy.stats import linregress
import os


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

def make_sst_trends_netcdf(modeldata,syear='1982-01-01'):
        ens=[]
        for i in modeldata.ensemble.values:
            print(i)
            model_tr=calc_longterm_trends(modeldata.sel(ensemble=i),syear=syear)
            ens.append(model_tr)

        model_tr=calc_longterm_trends(modeldata.mean(dim='ensemble'),syear=syear)
        ens.append(model_tr)
        trends=xr.concat(ens,dim='ensemble')
        return trends
         
def check_existing_file(spath,force=False):
    '''
    Returns False and deletes file if force is true
    Returns true if the path exists and can open. If corrupt it deletes and returns false.
    '''
    if os.path.isfile(spath)==True:
        if force==True:
            os.remove(spath)
            return False
        else:
            try:
                xr.open_dataset(spath)
                return True
            except:
                os.remove(spath)
                return False
    else:
        return False
    
    
    
def single_line_plotter(lmean,ltrend82,ltrend20,titles,ltrendmm=None,ltrendm=None,l_conversion=1,meancolormap='viridis',figsize=(10,8)):
    
    '''
    lmean     xrarr
    ltrend82  xrarr
    ltrend82p xarr
    ltrend20  xrarr
    ltrend20p pval xarr
    
    rmean     xrarr
    rtrend82  xrarr
    rtrend82p pval xarr
    rtrend20  xrarr 
    rtrend20p pval xarr
    titles    array[1,2,3,4,5,6]
    l_conversion float
    r_conversion float
    meancolormap     str cmap (ie viridis)
    
    Will dynamically produce a 3 x 2 (6) subplot with mean on top and 82 and 2000 trends below. 
    Can produce for any trend variable (produce mean over time, need a flag for this?)
    
    '''
    
    plt.figure(figsize=figsize)
    plt.subplot(311)
    if type(ltrendm)==type(None):
        (lmean.mean(dim='time')*l_conversion).plot(cmap=meancolormap) 
    else:
        (lmean.mean(dim='time')*l_conversion).plot(vmin=ltrendm[1],vmax=ltrendm[0],cmap=meancolormap)
    plt.title(titles[0])
    #(((cafe_co2_mean.stf10.mean(dim='time')/1000)*86400)*-12)
    #plt.title('CAFE ens mean mean CO2 flux out of ocean (gC m2 day)')

    plt.subplot(312)
    if type(ltrendmm)==type(None):
        (ltrend82*l_conversion).plot(cmap='bwr')
    else:
        (ltrend82*l_conversion).plot(vmax=ltrendmm[1],vmin=ltrendmm[0],cmap='bwr')
    plt.title(titles[1])
    #((((cafe_co2_82tr.trend/1000)*86400)*-12*1000)).plot(vmax=3,vmin=-3,cmap='bwr')#(vmin=-0.15,vmax=0.15,cmap='bwr')
    #plt.title('CAFE CO2 flux longterm trends 1982-2020  (mgC/m2/day/year)')
    #plt.contourf(cafe_co2_82tr.pval.lon,cafe_co2_82tr.pval.lat,cafe_co2_82tr.pval.values,colors='none',hatches=['.'],levels=[0,0.05])   
 


    plt.subplot(313)
    if type(ltrendmm)==type(None):
        (ltrend20*l_conversion).plot(cmap='bwr')
    else:
        (ltrend20*l_conversion).plot(vmax=ltrendmm[1],vmin=ltrendmm[0],cmap='bwr')
    plt.title(titles[2])
    #((((cafe_co2_20tr.trend/1000)*86400)*-12*1000)).plot(vmax=3,vmin=-3,cmap='bwr')#(vmin=-0.15,vmax=0.15,cmap='bwr')
    #plt.title('CAFE CO2 flux longterm trends 2000-2020  (mgC/m2/day/year)')
    #plt.contourf(cafe_co2_20tr.pval.lon,cafe_co2_20tr.pval.lat,cafe_co2_20tr.pval.values,colors='none',hatches=['.'],levels=[0,0.05])   
    #plt.tight_layout()

    plt.tight_layout()
    plt.show()
    
    

#THIS IS THE MAIN FUNCTION.. Still could have modifications to improve it. 

def CAFE60_eqpac_cutter(#modeldata_all,
                        modelType='BGC',
                        variable=None,
                        cut_eqpac=True,
                        save_all_data=False,
                        convert_times=True,
                        mean_of_ensemble=True,
                        fix_long_coords=True,
                        trend=False,
                        conversion=None,
                        ensemble_trends=False,
                        startday=1982,
                        endday=2020,
                        unit_fixer=False,
                        regridder=False,
                        plot=True,
                        force=False,
                        st_ocean=None,
                        raw_cafe_fp='/g/data/xv83/dcfp/CAFE60v1/',
                        processed_path='/g/data/xv83/np1383/processed_data/'):

    '''
    A large customisable function to slice the data we want out of the CAFE60 storage.
    Focuses on Monthly data. Could be expanded to process daily but not yet.
    Will not deal with depth (st_ocean variable) very well at the moment.  Could be added later.
    
    Save path  /scratch1/pit071/CAFE60/processed/*
    savename=Variable_Startday_TropPac_TR?_ENSMEM?

    modelType: BGC, Physics, Atmos
    variable = Either string or array of NAMES


    Mean_of_ensemble=True #To Ensemble #0. 
    Cut_eqpac= Default True (If False is global)
    convert_times=True #Convert to numpy datetime.
    fix_long_coords=Default False
    StartDay=1982
    EndDay=2019-01-01
    Trend=False (Calculate per pixel trends)
    Unit_fixer=False #Temporary. Maybe build this in later to automatically convert to useful units.
    Regridder=False #Temporary. For the moment do
    
    Conversion should include a modification to enact on datasets to convert to a desired format. For example *-12 for carbon to grams not moles. and then maybe * or /1000 for g/mg conversions.
    
    '''
    #Load the desired data in
    if modelType=='BGC':
        modeldata_all=xr.open_zarr(raw_cafe_fp+'ocean_bgc_month.zarr.zip',consolidated=True)
    elif modelType=='physics':
        modeldata_all=xr.open_zarr(raw_cafe_fp+'ocean_month.zarr.zip',consolidated=True)
    elif modelType=='atmos':
        modeldata_all=xr.open_zarr(raw_cafe_fp+'atmos_isobaric_month.zarr.zip',consolidated=True)
    
    #Grab the variable[s] we want.
    if type(variable)==type(None):
        print('No variable given')
        return False
    else:
        #Ok grab the variables out
        if type(variable)!=list:
            #Turn it into a list so we can iterate it
            variable=[variable]
        
        # -----
        #Ok this function loops through here.
        for var in variable:
            print('Starting on Variable: '+var)
            try:
                #This should work with one or more variables but will need to double check this works
                modeldata=modeldata_all[var]
            except:
                print('Incorrect variable given: '+var+ '\n Should be one of:\n'+str(list(modeldata_all.keys())))
                return False
    
            # --------------------------
            #Start modifying model data
            print('Begin Dataset Size: '+str(modeldata.nbytes/1e9) + ' GB')
        
            try:
                modeldata=modeldata.rename({'xt_ocean':'lon','yt_ocean':'lat'})
                print('renamed lons')
            except:
                #lats will be wrong way around
                print('vars probably already named')
      
            if fix_long_coords==True:
                modeldata['lon']=modeldata['lon']+360

            if cut_eqpac==True:
                modeldata=modeldata.sel(lon=slice(120,290),lat=slice(-20,20))
                region_name='eqpac/'
            else:
                region_name='global/'
  
            if convert_times==True:
                modeldata['time']=np.array(modeldata.indexes['time'].to_datetimeindex(), dtype='datetime64[M]')
                modeldata=modeldata.sel(time=slice(np.datetime64(str(startday)+'-01-01'),np.datetime64('2020-01-01')))
            else:
                print("Cut times for CF Not Implements. Try convert_times=True")

            if save_all_data==True:
                #Save the whole dataset here
                spath=savepath+region_name+str(var)+'_all_'+str(startday)+'.nc'
                if check_existing_file(spath,force)==False: #if it returns true then it exists and can open
                    print('Saving Dataset '+var+', size: '+str(modeldata.nbytes/1e9) + ' GB')
                    modeldata.to_netcdf(spath) 
                    print('saved to: '+spath)
                else:
                    print('Mean whole '+var+' Dataset already exists at: '+spath)
            
            st_ocean_marker=''
            if type(st_ocean)==list:
                print('Cut function currently does not support multiple depths')
                pass
                #sum or mean of the water column?
                #modeldata.sel(st_ocean=st_ocean).mean(dim='st_ocean')
                #modeldata.sel(st_ocean=st_ocean).sum(dim='st_ocean')
                
            elif type(st_ocean)==int:
                try:
                    modeldata=modeldata.sel(st_ocean=st_ocean)
                    st_ocean_marker='_'+str(st_ocean)+'m_'
                except:
                    print('Failed no st_ocean variable available try again, doing nothing.')
                
                
                
                
            if mean_of_ensemble==True:
                #SAVING ENS MEAN
                print('Calculating Ens Mean')
                spath=processed_path+'cafe/'+region_name+str(var)+'_ensmean_'+str(startday)+st_ocean_marker+'.nc'
                spath_std=processed_path+'cafe/'+region_name+str(var)+'_ensstd_'+str(startday)+st_ocean_marker+'.nc'
                
                if check_existing_file(spath,force)==False: #if it returns true then it exists and can open
                    if var=='stf10':
                         modeldata_mean=modeldata.sel(ensemble=1)
                    else:
                        modeldata_mean=modeldata.mean(dim='ensemble')
                        modeldata_std=modeldata.std(dim='ensemble')
                        
                    print('Saving Mean Ens Dataset Size: '+str(modeldata_mean.nbytes/1e9) + ' GB')
                    modeldata_mean.to_netcdf(spath) 
                    print('Saved to: '+spath)
                          
                    if check_existing_file(spath_std,force)==False:
                        modeldata_mean.to_netcdf(spath_std) 
                        print('Saved to: '+spath_std)
                    
                else:
                    print('Mean Ens Dataset already exists: '+spath)


            if trend==True:
                print('Calculating Trend')
                 #REQUIRES mean_of_ensemble=True
                modeldata_mean=xr.open_dataset(processed_path+'cafe/'+region_name+str(var)+'_ensmean_'+str(startday)+st_ocean_marker+'.nc')[var]

                spath82=processed_path+'cafe/'+region_name+str(var)+'_meantrends_'+str(1982)+st_ocean_marker+'.nc'
                spath20=processed_path+'cafe/'+region_name+str(var)+'_meantrends_'+str(2000)+st_ocean_marker+'.nc'

                if check_existing_file(spath82,force)==False:
                    trend_1982=calc_longterm_trends(modeldata_mean,startday='1982-01-01')
                    trend_1982.to_netcdf(spath82)
                else:
                    print('1982 trend Dataset already exists '+spath82)

                if check_existing_file(spath20,force)==False:
                    trend_2000=calc_longterm_trends(modeldata_mean,startday='2000-01-01')
                    trend_2000.to_netcdf(spath20)
                else:
                    print('2000 trend Dataset already exists: '+spath20)

                if plot==True:
                    print('plotting')
                    lmean=xr.open_dataset(spath)[var]
                    ltrend82=xr.open_dataset(spath82).trend
                    ltrend20=xr.open_dataset(spath20).trend
                    titles=[str(var)+' : mean',
                           str(var)+' : 1982-2020 trend',
                           str(var)+' : 2000-2020 trend']
                    single_line_plotter(lmean,ltrend82,ltrend20,titles)

            if ensemble_trends==True:
                print('Processing individual ensemble trends. Might take a while. Probably recommend only for eqpac')
                if len(startday)==4:
                    startday=str(startday)+'-01-01'
                trends=make_sst_trends_netcdf(modeldata,startday)
                spath=savepath+region_name+str(var)+'_enstrends_'+str(startday)+st_ocean_marker+'.nc'
                remove_existing_file(spath)
                trends.to_netcdf(spath)
                print('Saved to: '+spath)

    
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
        
    filepath=processed_path+'cafe/landshutzer'+'_'+savepath+'_regrid.nc'
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

            land_obs=xr.open_dataset(fp)
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


def lin_wrapper(obj,startyear=1982):
    '''
    #https://github.com/pydata/xarray/issues/1815
    #https://stackoverflow.com/questions/52094320/with-xarray-how-to-parallelize-1d-operations-on-a-multidimensional-dataset

    #This was just a test function to assess vectorised vs unvectorised version. This version takes approximately the same time as the looped version. 
    #Going to ignore this method.
    '''
    def new_linregress(x, y):
        # Wrapper around scipy linregress to use in apply_ufunc
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        #print(slope)
        return np.array([slope*365, p_value])

    obj=obj.where(obj!=-0.9999,np.nan)
    obj=obj.interpolate_na(dim='time').sel(time=slice(str(startyear)+'-01-01','2020-01-01'))
    obj['time']=pd.to_numeric(obj.time.values.astype('datetime64[D]'))
    
    stat = xr.apply_ufunc(new_linregress, obj['time'], obj,
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[['stats']],
                           vectorize=True,
                           dask="parallelized",
                           output_dtypes=['float64'],
                           output_sizes={"stats": 2})
    return stat

if __name__ == '__main__':
    pass