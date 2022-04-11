import pandas as pd
import xarray as xr
import numpy as np
from scipy import stats
#from scipy.stats import linregress
#xarray_detrend
#xarray_#get_trend
def xarray_detrend(dat,keep_intercept_values=True,return_trend=False,input_core_dims='time'):
    '''
    For example, if you have a simple linear trend for the mean, calculate the least squares regression 
    line to estimate the growth rate, r. Then subtract the deviations from the least squares fit line 
    (i.e. the differences) from your data.
    https://www.statisticshowto.com/detrend-data
    '''
    def lin_xarray_detrender(x,y,keep_intercept_values=True,return_trend=False):
        x=pd.to_numeric(x.astype('datetime64[D]'))
        # Wrapper around scipy linregress to use in apply_ufunc
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        m=slope
        b=intercept
        if return_trend==True:
            return (m*x+b)
        if keep_intercept_values==True:
            # Check intercept x=0. Use mean / median otherwise.
            return b+(y-(m*x + b))
        elif keep_intercept_values==False:
            return y-(m*x + b)
        # Needs to happen inside this function.
        # Ok actually just use the linregress function above and thenb 

    x=dat[input_core_dims]
    y=dat
    #stat=xarray_get_trend(dat,input_core_dims)
    #m=stat.sel(parameter=0) #slope
    #b=stat.sel(parameter=1) #intercept
    #detrend_y=b+(y-(m*x + b))
    # This doesnt seem to work because of shapes.... sure theres a way to broadcast the dimensions together but meh.
    
    detrended = xr.apply_ufunc(lin_xarray_detrender, x,y,keep_intercept_values,return_trend,
                       input_core_dims=[[input_core_dims], [input_core_dims],[],[]],
                       output_core_dims=[[input_core_dims]],
                       vectorize=True,
                       dask="parallelized",
                       output_dtypes=['float64'],
                       output_sizes={input_core_dims:len(x)})
    
    return detrended

# return a new DataArray
def xarray_get_trend(dat,input_core_dims='time'):
    '''
    x is time
    y is desired variable
    
    expected time, otherwise input_core_dims='
    Returns Paramaters
    #parameters are [slope, intercept, r_value, p_value, std_err]
    '''
    
    def lin_xarray_linregress(x,y):
        x=pd.to_numeric(x.astype('datetime64[D]'))
        # Wrapper around scipy linregress to use in apply_ufunc
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return np.array([slope, intercept, r_value, p_value, std_err])
    
    x=dat[input_core_dims]
    y=dat
    stat = xr.apply_ufunc(lin_xarray_linregress, x,y,
                          input_core_dims=[[input_core_dims], [input_core_dims]],
                          output_core_dims=[["parameter"]],
                          vectorize=True,
                          dask="parallelized",
                          output_dtypes=['float64'],
                          output_sizes={"parameter": 5})
    return stat

# This method doesnt do nans and also doesnt keep relative magnitude
#def detrender(da):
#    return xr.apply_ufunc(
#            sps.detrend,
#            da.fillna(-9999),
#            1,
#            'constant',
#            output_dtypes=[da.dtype],
#            dask="parallelized",
#            )
def deseasonaliser(da,keep_relative=True,return_seasonality=False,rolling=False):
    if rolling==True:
        return da.rolling(dim={'time':12}).mean()
    deseasonalised= da.groupby("time.month") - da.groupby("time.month").mean("time")
    seasonality=(da-deseasonalised)
    base_seasonality=seasonality-seasonality.mean(dim='time')
    detrended_season_removed=da-base_seasonality
    if keep_relative==True:
        returns=[detrended_season_removed]
    else:
        returns=[deseasonalised]
    return returns[0]


 #(dim='time')



if __name__ == '__main__':
    pass