import requests                      #Version '2.19.1'
import xarray as xr                  #Version '0.11.3'
import numpy as np                   #Version '1.16.1' (Not used here, but in chl_tpca_algorithms)
import matplotlib.pyplot as plt      #Version '3.0.0'
import os
from bs4 import BeautifulSoup
import multiprocessing
import sys
import pandas as pd
from tarfile import TarFile
import gzip
import subprocess
import calendar
import shutil

def gzipper(inf,outf):
    try:
        inp = gzip.GzipFile(inf, 'rb')
        s = inp.read()
        inp.close()
        outp = open(outf, 'wb')
        outp.write(s)
        outp.close()
        return outf
    except:
        return False
    

    
def downloader(urls,sensor='viirs',ppname='vgpm'):
    """" Function designed to download satellite data from http://orca.science.oregonstate.edu/"""
    path='datasets/npp_satellite/'+ppname+'_'+sensor
    path1=path+'_extracted'
    path2=path+'_opened'
    path3=path+'_converted'
    path4=path+'_nc'
    
    def exists(fileloc):
        #print(fileloc)
        try:
            tar=TarFile.open(fileloc)
            tar.extractall(path1)
            return True
        except:
            return False
            
        #TarFile.extractall(tar)

        #try:
        #    
        #    return True
        #except:
        #    return False
    if not os.path.isdir('datasets'):
        print('Creating directory: ','datasets')
        os.makedirs('datasets')  
    if not os.path.isdir('datasets/npp_satellite/'):
        print('Creating directory: ','datasets/npp_satellite/')
        os.makedirs('datasets/npp_satellite/')  

    if not os.path.isdir(path):
        print('Creating directory: ',path)
        os.makedirs(path)    
    if not os.path.isdir(path1):
        print('Creating directory: ',path1)
        os.makedirs(path1)    
    if not os.path.isdir(path2):
        print('Creating directory: ',path2)
        os.makedirs(path2)  
        
    if not os.path.isdir(path3):
        print('Creating directory: ',path3)
        os.makedirs(path3)  
        
    if not os.path.isdir(path4):
        print('Creating directory: ',path4)
        os.makedirs(path4)     
        
    file_locations=[]
    count=0
    for url in urls:
        while True:
        #Download the files to their file name in the directory we just created.
        #ie: seawifs_data/S2000001.L3m_DAY_RRS_Rrs_443_9km.nc
            try:
                fileloc=path+'/'+url.split('/')[-1]
            except:
               # print('something broke at:',url)
                continue
            print(url)
            if exists(fileloc):
                print('Exists: ',fileloc)
                file_locations.append(fileloc)
                break
            r = requests.get(url)#,timeout=s20)
            with open(fileloc, 'wb') as f:
                f.write(r.content)
    
            #Ensure that the file actually downloaded, this can fail sometimes for some reason, maybe too soon.
            #time.sleep(1) #Can fail sometimes so maybe this is a fix
            if (r.status_code==200) & (exists(fileloc)==True):
                print('Downloaded: ',fileloc)
                file_locations.append(fileloc)
                break
            else:
                print('Download failed:', fileloc,'status:',r.status_code)
                count+=1
                if count>=10:
                    break
    return file_locations


def unzip_extract_convert(sensor='viirs',ppname='vgpm'):
    print('Unzipping the bz')
    fp='datasets/npp_satellite/'+ppname+'_'+sensor
    for f in os.listdir(fp+'_extracted'):
        gzipper(fp+'_extracted/'+f,fp+'_opened/'+f[0:-3])
    #sh fly_converter.sh vgpm_sw_opened/vgpm.2012092.hdf vgpm_sw_converted/test.hd5
    
    #print('Use h4toh5 binary software to convert, change the vars around and save as nc')
    #print('h4toh5 tool located at: https://support.hdfgroup.org/products/hdf5_tools/h4toh5/download.html')
    #Located from https://support.hdfgroup.org/products/hdf5_tools/h4toh5/download.html
    for f in os.listdir(fp+'_opened'):
        fname=f
        
        #Older version of xarray needed h4toh5 but this seems to work now in updated version.
        #subprocess.run(["./h4toh5",fp+'_opened/'+f,fp+'_converted/'+fname])
        #print('Opening : ' +str(f))
        dat=xr.open_dataset(fp+'_opened/'+fname) #Pynio was needed but maybe no more,engine='pynio')
        dat=dat.rename({'fakeDim0':'lat','fakeDim1':'lon'})
        #dat['lat']=np.arange(-90,90,180/len(dat.lat))*-1
        #dat['lon']=np.arange(-180,180,360/len(dat.lon))
        dat['lat']=np.linspace(-90,90,len(dat.lat))*-1
        dat['lon']=np.linspace(-180,180,len(dat.lon))
        #print(dat.attrs)
        
        dat['time']=np.datetime64(dat.attrs['Start Time String'][6:10]+'-'+dat.attrs['Start Time String'][0:2]+'-'+dat.attrs['Start Time String'][3:5])
        dat=dat.assign_coords(lon=(dat.lon % 360)).roll(lon=(dat.dims['lon'] // 2), roll_coords=True)	
        dat=dat.sel(lat=slice(20,-20))
        dat=dat.sel(lon=slice(120,290))
        dat=dat.assign_coords({'time':dat.time})
        #print(final_dat)
        # outp=sensor+'_'+ppname+'_'+dat.attrs['Start Time String'][6:10]+'-'+dat.attrs['Start Time String'][0:2]+'-'+dat.attrs['Start Time String'][3:5]
        dat.to_netcdf('datasets/npp_satellite/'+ppname+'_'+sensor+'_nc/'+sensor+'_'+f[0:-3]+'nc')
        
    print('Deleting unneccessary Data...')
    shutil.rmtree(r'datasets/npp_satellite/'+ppname+'_'+sensor+'_extracted', ignore_errors=True)
    shutil.rmtree(r'datasets/npp_satellite/'+ppname+'_'+sensor+'_opened', ignore_errors=True)
    shutil.rmtree(r'datasets/npp_satellite/'+ppname+'_'+sensor+'_converted', ignore_errors=True)
    

def Download_TPCA():
    '''
    Function to download TPCA MODIS and SeaWIFS from nci.org.
    
    Possible to also use DAPPS through xarray and save the files
    rather than using requests.
    '''
    path_sw='/g/data/xv83/np1383/external_data/chl/TPCA/seawifs/'
    if not os.path.isdir(path_sw):
        print('Creating directory: ',path_sw)
        os.makedirs(path_sw)
       
    path_mod='/g/data/xv83/np1383/external_data/chl/TPCA/modis/'
    if not os.path.isdir(path_mod):
        print('Creating directory: ',path_mod)
        os.makedirs(path_mod)
    
    
    tpca_link=['http://dapds00.nci.org.au/thredds/fileServer/ks32/CLEX_Data/TPCA_reprocessing/v2019_01/']
    sensors=['SeaWiFS/tpca_seawifs_','MODIS-Aqua/tpca_modis_aqua_'] #and then year
    sens=['sw','mod']           
    #Download SeaWiFS files from the above array, spaced by each year.
    
    for i in range(0,2): #To do SeaWiFS and then MODIS
        for yr in np.arange(1997,2020):
            if i==0: #SW
                sensor=tpca_link[0]+sensors[0]+str(yr)+'.nc'
                path=path_sw
            elif i==1: #MODIS
                sensor=tpca_link[0]+sensors[1]+str(yr)+'.nc'
                path=path_mod
        
            #Start the download
            try:
                r = requests.get(sensor)#,timeout=s20)
                fileloc=path+sensors[0].split('/')[1]+str(yr)+'.nc'
                if r.status_code!=404:
                    with open(fileloc, 'wb') as f:
                        f.write(r.content)
                    print('Downloaded: ' + sens[i] + str(yr))
                else:
                    print(i,str(r.status_code))
            except KeyboardInterrupt:
                import sys
                sys.exit()
            except:
                print(str(yr)+ sens[i]+'  Unavailable')
            pass