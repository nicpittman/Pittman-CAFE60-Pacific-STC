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

import os
os.chdir('/g/data/xv83/np1383/src_CAFE60_eqpac_analysis_code/')

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
# -

pwd

from C60_model_cutting_functions import CAFE60_eqpac_cutter

# This scipt will load a Dask Scheduler, and then call the CAFE60 model cutting functions for different varaibles. Those functions are stored in `a_model_cutting_functions.py`. This should be easy enough to make a new variable cutout by just copying the function templates.
#
# Current variables bing cutout include:
#
# sst
# stf10 (co2 flux)
# phy (phytoplankton?)
# surface_zoo
# surface_dic
# stf10 (Natural + Anth CO2?)
# stf03 (Not sure which is which)
# stf07 (Not sure which is which).
# pprod_gross_2d
# export_prod
#
#
# Easy to add others and make a bit more modular. Will need some work to get the st_ocean depths out but its kind of there. 

# +
# Set up the Local Dask Cluster
# from dask.distributed import Client, LocalCluster,Scheduler
# from dask_jobqueue import SLURMCluster
# cluster = LocalCluster()
# client = Client(cluster)
# print(client)
# -

# Set up the remote dask cluster
from dask.distributed import Client,Scheduler
from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(cores=2,memory="16GB")
client = Client(cluster)
cluster.scale(cores=8)
#cluster.adapt(minimum=2, maximum=16)
client
##

# +
#cluster.scale(cores=2)
# -

client



# !squeue -u np1383





# +
# Check the model data works first

# modeldata_bgc=xr.open_zarr('/g/data/xv83/dcfp/CAFE60v1/ocean_bgc_month.zarr.zip',consolidated=True)
# modeldata_ocean=xr.open_zarr('/g/data/xv83/dcfp/CAFE60v1/ocean_month.zarr.zip',consolidated=True)
# modeldata_atmos=xr.open_zarr('/g/data/xv83/dcfp/CAFE60v1/atmos_isobaric_month.zarr.zip',consolidated=True)
# -

#Process CAFE SST for equatorial Pacific
CAFE60_eqpac_cutter(modelType='physics',
                        variable='sst',
                        cut_eqpac=False,
                        save_all_data=False,
                        trend=True,
                        force=True)

#Process CAFE SST for equatorial Pacific
CAFE60_eqpac_cutter(modelType='physics',
                        variable='sst',
                        cut_eqpac=True,
                        save_all_data=True,
                        trend=True,
                        force=True)

#cluster.scale(cores=8)
CAFE60_eqpac_cutter(modelType='BGC',
                        variable=['stf03','stf07','stf10'],
                        cut_eqpac=True,
                        save_all_data=True,
                        trend=True,
                        plot=False,
                        force=True)
#cluster.scale(cores=2)

CAFE60_eqpac_cutter(modelType='BGC',
                        variable='surface_phy',
                        cut_eqpac=True,
                        save_all_data=False,
                        trend=True,
                        plot=False,
                        force=True)



CAFE60_eqpac_cutter(modelType='BGC',
                        variable='phy',
                        cut_eqpac=True,
                        save_all_data=False,
                        trend=True,
                        plot=False,
                        force=True,
                        st_ocean=15)

CAFE60_eqpac_cutter(modelType='BGC',
                        variable=['surface_dic','surface_zoo'],
                        cut_eqpac=True,
                        save_all_data=False,
                        trend=True,
                        plot=True,
                        force=True)

client

CAFE60_eqpac_cutter(modelType='BGC',
                        variable=['pprod_gross_2d','export_prod'],
                        cut_eqpac=True,
                        save_all_data=False,
                        trend=True,
                        plot=True,
                        force=False)

CAFE60_eqpac_cutter(modelType='atmos',
                        variable=['u_ref','v_ref'],
                        cut_eqpac=True,
                        save_all_data=False,
                        trend=True,
                        fix_long_coords=False,
                        plot=True,
                        force=False)

# Ok and Calculate and save the windspeed
u=xr.open_dataset('/g/data/xv83/np1383/processed_data/cafe/eqpac/u_ref_ensmean_1982.nc')
v=xr.open_dataset('/g/data/xv83/np1383/processed_data/cafe/eqpac/v_ref_ensmean_1982.nc')
ws=np.sqrt((u.u_ref**2)+(v.v_ref**2))
ws.to_netcdf('/g/data/xv83/np1383/processed_data/cafe/eqpac/ws_ensmean_1982.nc')


