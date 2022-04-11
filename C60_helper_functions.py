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
    
    
def lin_wrapper(obj,startyear=1982,endyear=2020):
    '''
    #https://github.com/pydata/xarray/issues/1815
    #https://stackoverflow.com/questions/52094320/with-xarray-how-to-parallelize-1d-operations-on-a-multidimensional-dataset

    #This was just a test function to assess vectorised vs unvectorised version. This version takes approximately the same time as the looped version. 
    #Going to ignore this method.
    Returns current units / year. 
    '''
    def new_linregress(x, y):
        # Wrapper around scipy linregress to use in apply_ufunc
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        #print(slope)
        return np.array([slope*365, p_value])

    obj=obj.where(obj!=-0.9999,np.nan)
    obj=obj.interpolate_na(dim='time').sel(time=slice(str(startyear)+'-01-01',str(endyear)+'-01-01'))
    obj['time']=pd.to_numeric(obj.time.values.astype('datetime64[D]'))
    
    stat = xr.apply_ufunc(new_linregress, obj['time'], obj,
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[['stats']],
                           vectorize=True,
                           dask="parallelized",
                           output_dtypes=['float64'],
                           output_sizes={"stats": 2})
    return stat




def find_enso_events(threshold=0.5,data_path='../external_data/indexes/',out_path='../processed_data/indexes/'):
    '''
    A function to pull ENSO data from our datasets/indexes/meiv2.csv
    save events (months) stronger than threshold (0.5 by default)
    
    Modified to include CP, EP, El Nino and La Nina events and are saved to csv.
    
    'processed/indexes/el_nino_events.csv'
    'processed/indexes/la_nina_events.csv'
    'processed/indexes/ep_el_nino_events.csv'
    'processed/indexes/cp_el_nina_events.csv'
    
    Returns
    -------
    None.
    
    '''
    
    #enso=pd.read_csv('datasets/indexes/meiv2.csv',index_col='Year')
    enso=pd.read_csv(data_path+'meiv2.csv',index_col=0,header=None)
    enso=enso.iloc[3:] #Just so Both EMI and MEI start in 1981-01-01
    enso_flat=enso.stack()
    enso_dates=pd.date_range('1982','2021-10-01',freq='M')- pd.offsets.MonthBegin(1) #Probably want to check this is correct if updating.
    
    emi=pd.read_csv(data_path+'emi.csv')
    emi.time=emi.Date.astype('datetime64[M]')
    emi.index=emi.time
    emi=emi.EMI
    
    enso_timeseries=pd.DataFrame({'Date':enso_dates,'mei':enso_flat})
    
    
    #Check if we are in or out of an event so far
    el_event=False
    la_event=False
    ep_event=False
    cp_event=False
    cpc_event=False
    el_startdate=''
    la_startdate=''
    ep_startdate=''
    cp_startdate=''
    cpc_startdate=''
    
    elnino=pd.DataFrame()
    lanina=pd.DataFrame()
    cp=pd.DataFrame()
    cpc=pd.DataFrame()
    ep=pd.DataFrame()
    
    month_threshold=5 #Months over threshold)
    threshold=0.5
    
    #All El Nino
    for i,today in enumerate(enso_timeseries.Date):
        val=enso_timeseries.mei.iloc[i]
        if val>=threshold:
            if el_event==False:  #And we havent yet entered an event
                el_startdate=today
                el_event=True
            else:
                pass
                #Dont need to do anything because it will get caught later
        else:
            if el_event==True:
                if ((today-el_startdate)>=np.timedelta64(month_threshold,'M')): #Make sure event is long enough
         
                    if el_startdate.to_datetime64()!=enso_timeseries.Date.iloc[i-1].to_datetime64():
                        elnino=elnino.append({'start':el_startdate.to_datetime64(),
                                              'end':enso_timeseries.Date.iloc[i-1],
                                              'mei':enso_timeseries.mei.iloc[i-1]},ignore_index=True)
                        el_event=False
                else: el_event=False
    
    #La Nina
    for i,today in enumerate(enso_timeseries.Date):
        val=enso_timeseries.mei.iloc[i]
        if val<=-threshold:
            if la_event==False:  #And we havent yet entered an event
                la_startdate=today
                la_event=True
            else:
                pass
                #Dont need to do anything because it will get caught later
        else:
            if la_event==True:
                if ((today-la_startdate)>=np.timedelta64(month_threshold,'M')): #Make sure event is long enough
         
                    if la_startdate.to_datetime64()!=enso_timeseries.Date.iloc[i-1].to_datetime64():
                        
                        lanina=lanina.append({'start':la_startdate.to_datetime64(),
                                          'end':enso_timeseries.Date.iloc[i-1],
                                          'mei':enso_timeseries.mei.iloc[i-1]},ignore_index=True)
                        la_event=False
                else: la_event=False
        
    
    #CP events
    for i,today in enumerate(emi.index):
        #val=emi.iloc[i]
        val=np.mean(emi.iloc[i:i+2])
        if val>=threshold:
            if cp_event==False:  #And we havent yet entered an event
                cp_startdate=today
                cp_event=True
            else:
                pass
                #Dont need to do anything because it will get caught later
        else:
            if cp_event==True:
                if ((today-cp_startdate)>=np.timedelta64(month_threshold,'M')): #Make sure event is long enough
                    if cp_startdate.to_datetime64()!=emi.index[i-1].to_datetime64():
                        cp=cp.append({'start':cp_startdate.to_datetime64(),
                                          'end':emi.index[i-1],
                                          'emi':emi.values[i-1]},ignore_index=True)
                        cp_event=False
                else: cp_event=False
        
    
    #EP El Nino
    for i,today in enumerate(enso_timeseries.Date):
        val=enso_timeseries.mei.iloc[i]
        val1=np.mean(enso_timeseries.mei.iloc[i])
        try:
            emi_val=emi.iloc[i]
            emi_val1=np.mean(emi.iloc[i:i+8]) #Just to make sure the 2015 EP event is classified as such
        except IndexError as ie:
            # FUTURE WARNING UPDATE EMI DATASET!
            #print('Err: '+str(ie))
            emi_val=0
            emi_val1=0
        #print(emi_val)
        #print(today)
        #print(emi.index[i])
        #print(enso_timeseries.iloc[i].Date)
        #print()
        #print(emi_val,val)
        #print('\n')
        
        #print()
        if (val1>=threshold)&(emi_val1<threshold):#&(emi_val1<threshold):
            if ep_event==False:  #And we havent yet entered an event
                ep_startdate=today
                ep_event=True
            else:
                pass
                #Dont need to do anything because it will get caught later
        else:
            if ep_event==True:
                if ((today-ep_startdate)>=np.timedelta64(month_threshold,'M')): #Make sure event is long enough
             
                    if ep_startdate.to_datetime64()!=enso_timeseries.Date.iloc[i-1].to_datetime64():
                        ep=ep.append({'start':ep_startdate.to_datetime64(),
                                          'end':enso_timeseries.Date.iloc[i-1],
                                          'mei':enso_timeseries.mei.iloc[i-1]},ignore_index=True)
                        ep_event=False
                else: ep_event=False
    
    
    #print(elnino)
    #print(lanina)
    #print(cp)
    #print(ep)
    
    elnino.to_csv(out_path+'el_nino_events.csv')
    lanina.to_csv(out_path+'la_nina_events.csv')
    cp.to_csv(out_path+'cp_events.csv')
    ep.to_csv(out_path+'ep_events.csv')

    print('saved to: '+out_path+'el_nino_events.csv')
    print('saved to: '+out_path+'la_nino_events.csv')
    print('saved to: '+out_path+'cp_events.csv')
    print('saved to: '+out_path+'ep_events.csv')
    


    

from datetime import datetime as dt
def decimalYearGregorianDate(date, form="datetime"):
    """
    Function copied from: https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/decimalYear.py
    
    Convert decimal year into gregorian date.
    
    Formally, the precision of the result is one microsecond.
    
    Parameters
    ----------
    date : float
        The input date (and time).
    form : str, optional
        Output format for the date. Either one of the following strings
        defining a format: "dd-mm-yyyy [hh:mm:ss]", "yyyy-mm-dd [hh:mm:ss]",
        where the term in braces is optional, or 
        "tuple" or "datetime". If 'tuple' is specified, the result will be a
        tuple holding (year, month, day, hour, minute, second, microseconds).
        In the case of "datetime" (default), the result will be a
        datetime object.
      
    Returns
    -------
    Gregorian date : str, tuple, or datetime instance
        The gregorian representation of the input in the specified format.
        In case of an invalid format, None is returned.
    """
  
    def s(date):
        # returns seconds since epoch
        return (date - dt(1900,1,1)).total_seconds()

  # Shift the input of 1e-2 microseconds
  # This step accounts for rounding issues.
    date += 1e-5/(365.*86400.0)
    
    year = int(date)
    yearFraction = float(date) - int(date)
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)
    secondsInYear = (s(startOfNextYear) - s(startOfThisYear)  ) * yearFraction
    # Find the month
    m = 1
    while m<=12 and s(dt(year=year, month=m, day=1)) - s(startOfThisYear) <= secondsInYear: m+=1
    m-=1
    # Find the day
    d = 1
    tdt = dt(year=year, month=m, day=d)
    while s(tdt) - s(startOfThisYear) <= secondsInYear:
      d+=1
      try: tdt=dt(year=year, month=m, day=d)
      except: break
    d-=1 
    # Find the time
    secondsRemaining = secondsInYear + s(startOfThisYear) - s(dt(year=year, month=m, day=d))
    hh = int(secondsRemaining/3600.)
    mm = int((secondsRemaining - hh*3600) / 60.)
    ss = int(secondsRemaining - hh*3600 - mm * 60) 
    ff = secondsRemaining - hh*3600 - mm * 60 - ss
    
    # Output formating
    if "tuple" == form:
      r = (year, m, d, hh, mm, ss, int(ff*1000))
    elif "datetime" == form:
      r = dt(year, m, d, hh, mm, ss, int(ff*1000))
    elif "dd-mm-yyyy" in form:
      r = str("%02i-%02i-%04i" % (d,m,year))
      if "hh:mm:ss" in form:
        r+=str(" %02i:%02i:%02i" % (hh,mm,ss))
    elif "yyyy-mm-dd" in form:
      r = str("%04i-%02i-%02i" % (year,m,d))
      if "hh:mm:ss" in form:
        r+=str(" %02i:%02i:%02i" % (hh,mm,ss))
    else:
      raise(ValueError("Invalid input form of `form` parameter.", \
                           where="gregorianDate"))
      return None
    return r      
    
    

def convert_CAFEatm_CO2times(atm_co2):
    '''
    Basically just wraps the above function into the format we want.
    '''
    dates=(atm_co2.time+1700)
    new_dates=[]
    for t in dates:
        new_t=np.datetime64(decimalYearGregorianDate(t)).astype('datetime64[M]')#.astype('datetime64[M]')
        new_dates.append(new_t)

    atm_co2['time']=new_dates
    return atm_co2


def solubility(tk,s):
    '''
    Calculate CO2 solubility depending on temperature (Kelvin) and salinity
    Uses coefficients provided in Wanninkhof et al., (2014) and Weiss (1974)
    Parameters
    ----------
    tk : Int or Float
        Temperature in Kelvin (Degrees C +273.15)
    s : Int or Float
        Salinity in Practical Salinity Units (g/kg)        
    Returns
    -------
    ko : float
        Ko is the solubility of CO2 in seawater, given temperature and salinity.
    '''
    A1=-58.0931
    A2=90.5069
    A3=22.2940
    B1=0.027766
    B2=-0.025888
    B3=0.0050578

    ko=np.exp(A1+ A2 *(100/tk) + A3 *np.log(tk/100) + s*(B1 + B2 *(tk/100) + B3 *(tk/100)**2))

    return ko




if __name__ == '__main__':
    pass