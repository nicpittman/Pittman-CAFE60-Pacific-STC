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

from C60_helper_functions import check_existing_file, single_line_plotter

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
                    st_ocean_marker='_'+str(st_ocean)+'m'
                except:
                    print('Failed no st_ocean variable available try again, doing nothing.')
                
            if save_all_data==True:
                #Save the whole dataset here
                spath=processed_path+'cafe/'+region_name+str(var)+'_ensmean_'+str(startday)+st_ocean_marker+'_all_ensembles.nc'
                if check_existing_file(spath,force)==False: #if it returns true then it exists and can open
                    print('Saving Dataset '+var+', size: '+str(modeldata.nbytes/1e9) + ' GB')
                    modeldata.to_netcdf(spath) 
                    print('saved to: '+spath)
                else:
                    print('Mean whole '+var+' Dataset already exists at: '+spath)
            
                
                
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

    

if __name__ == '__main__':
    pass