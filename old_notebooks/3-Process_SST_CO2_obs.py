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
#     display_name: Python [conda env:analysis3]
#     language: python
#     name: conda-env-analysis3-py
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
# Lets import our functions. There are no tests here. Stored separately so as to not make a mess.
# Make sure in right dir first

os.chdir('/g/data/xv83/np1383/src_CAFE60_eqpac_analysis_code/')
from C60_obs_functions import cut_regrid_reynolds_sst,cut_process_sst_obs_trends,proc_landschutzer,process_co2_land_trends, proc_rodenbeck,process_co2_rodenbeck_trends
from C60_helper_functions import check_existing_file
# -



# Set up the remote dask cluster
from dask.distributed import Client,Scheduler
from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(cores=2,memory="16GB")
client = Client(cluster)

cluster.scale(cores=8)

cluster



#Cut out global and eqpac reynolds SST and calculate trends for them. Saved in /scratch1/pit071/CAFE60/processed/obs/
#cuttropics=False,force=False,
cut_regrid_reynolds_sst(True,True) #Cut eqpac SST,force (If force=True will delete any existing file and resave)
cut_regrid_reynolds_sst(False,True) #Cut global SST and force save 
cut_process_sst_obs_trends(True) #Force trends for both global and eqpac. Should do both eqpac and global.  

# +
#Basically the same as above but for landshutzer.
#proc_landschutzer(False,True) 

#cuttropics=False,force=False,
proc_landschutzer(True,True)
proc_landschutzer(False,True) 
process_co2_land_trends(True) #Force trends for both global and eqpac. Should do both eqpac and global.  
# -



# Ok also do the rodenbeck CO2 fluxcor product.
proc_rodenbeck(cuttropics=True,force=True)
proc_rodenbeck(cuttropics=False,force=True)
process_co2_rodenbeck_trends(force=True)


