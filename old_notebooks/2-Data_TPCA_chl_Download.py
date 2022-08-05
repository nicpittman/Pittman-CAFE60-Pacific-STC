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

from downloader import download_TPCA

# Downloads TPCA chl Data from Research Data Portal. 

# +
# Set up the remote dask cluster
from dask.distributed import Client,Scheduler
from dask_jobqueue import SLURMCluster
cluster = SLURMCluster(cores=2,memory="16GB")
client = Client(cluster)
#cluster.scale(cores=4)
cluster.scale(cores=4)#adapt(minimum=2, maximum=10)

##
# -

cluster



download_TPCA()



# +
# REGRID
# -








