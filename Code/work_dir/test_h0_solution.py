#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:35:03 2023

@author: icd
"""



#%% Modules
import sys
import getopt
import json
import pickle
import os
import os.path as op
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import PersistenceImages
from scipy.stats import norm
import matplotlib.pyplot as plt
import persim

#IN-HOUSE IMPORTS
sys.path.append('/Users/icd/Dropbox/Projects/RJF/REFERRAL_NETWORK_TOPOLOGY_SHARING/analysis/felix/ML_physicians_network/Code/modules/')
import cv_prep_vars


###
#
#   Main
#
###

##
#%%   COLLECT INPUT
##

##
# Required Args
##
# rfr: RandomForestRegression, lasso: Lasso, en: Elastic Net
algorithm_to_use = ''
year             = '' #2011
# Select outcome column name to use
outcome_to_use   = '' #ar_stdprice_total
# Select pixel resolution for PIs: small, mid, large
pixel_opt        = ''

#TESTING
algorithm_to_use = 'rfr'
year = '2011'
outcome_to_use = 'ar_stdprice_total'
pixel_resolution = cv_prep_vars.pixel_resolution_map['mid']; pixel_opt = 'mid'

##
# Opt Args
#%% Set Defaults
##
# Decide on which dim to use
hdim              = '0'
# Select Metric for Scoring:
metric_str        = 'mse'
# Number of folds to use
k                 = 5
# Percent of sample to use for testing
test_size_percent = 0.33
# Seed to use or generate
seed_opt          = False
original_seed     = 3667424171
# Overwrite previous saved info (will not try to start from where it left off)
overwrite_opt     = False

# Misc vars
outcome_col_y = 1


#%% Generate Seed
if type(seed_opt) == int:
    seed_generated = seed_opt
elif not seed_opt:
    seed_generated = original_seed
else:
    seed_generated = np.random.randint(2**32)


#%% Filenames
opts_info = {
    "algorithm"  : algorithm_to_use,
    "year"       : year,
    "outcome"    : outcome_to_use,
    "resolution" : pixel_opt,
    "hdim"       : hdim,
    "metric"     : metric_str,
    "k"          : k,
    "test_prc"   : test_size_percent,
    "seed"       : str(seed_generated)
    }


# To load PDs
data_pds_fn_template = "proj-PI_year-{year}_region-great_lakes_desc-PD_as_str_h1.csv"

# To load KFolds indices
cv_prep_fn_template = "proj-PI_year-{year}_region-great_lakes_k-{k}_" + \
                    "test-{test_prc}_seed-{seed}_desc-cv_prep_data.json"

# To save output from this script
savename_template = "cv_alg-{algorithm}_outcome-{outcome}_year-{year}_region-great_lakes" +\
                    "_res-{resolution}_hdim-{hdim}_metric-{metric}_seed-{seed}_desc-"

script_name = savename_template.format_map(opts_info)

output_fn          = op.join(cv_prep_vars.CV_OUTPUT, script_name + 'cv_results.txt')
last_save_state_fn = op.join(cv_prep_vars.CV_OUTPUT, script_name + 'last_save_state.pickle')


#%% Outcome
# Load outcome data
outcome_df = cv_prep_vars.get_outcome_df(outcome_to_use)


#%% Load PD data
data_pds_fn = op.join(cv_prep_vars.DATA_PATH, data_pds_fn_template.format_map(opts_info))
data_pds    = pd.read_csv(data_pds_fn)

# change nas to '' to help with the conversion
data_pds.fillna('', inplace = True)


#%% Set metric to use
if metric_str.lower() == "mse":
    metric_to_use = metrics.mean_squared_error
elif metric_str.lower() == "mae":
    metric_to_use = metrics.mean_absolute_error
elif metric_str.lower() == "r2":
    metric_to_use = metrics.r2_score


#%% Prep PI vars
pixel_resolution_info = cv_prep_vars.get_resolution_info(pixel_resolution)

pixel_resolution_val = pixel_resolution_info['resolution']
img_shape            = pixel_resolution_info['img_shape']
vec_len              = pixel_resolution_info['vec_len']

# These columns are used to extract the PI columns from the data.
# They are always in the location of the h0 columns.
hdim_cols = pixel_resolution_info['h0_cols']


#%%(1) transforming the death values into the persistence value (death - birth)
# data_pds.h0: each row is HSA's list of holes as a string
#   birth-death;birth-death;...

# List of birth-death values a np array: [[birth, death], ...]
# This is what I submit to the function to generate the PI
pers_dgm = np.array([[float(e2) for e2 in elem.split('-')] for elem in data_pds.h0[0].split(';')])

#   WHAT IS RUN AFTER
# Convert to  PI
#PI = pers_imgr.transform(diagram, skew=True)
# Convert to vec and return
#return PI.reshape((PI.size))

#%% Using the transform func in the orig package
skew = True


# pers_imgr: for testing self commands

pers_dgm = np.copy(pers_dgm)
pers_img = np.zeros((20,))
# OK, so this shape is determined by the resolution...but that's for 2D
# let's try (20,)

n = pers_dgm.shape[0]
general_flag = True


if skew:
    pers_dgm[:, 1] = pers_dgm[:, 1] - pers_dgm[:, 0]

# compute weights at each persistence pair
#wts = self.weight(pers_dgm[:, 0], pers_dgm[:, 1], **self.weight_params)
# OK, so here we had decided to use the linear ramp, so self.weight == linear ramp
wgt_fx = PersistenceImages.weighting_fxns.linear_ramp
wts = wgt_fx(pers_dgm[:, 0], pers_dgm[:, 1], end = 1.01)

# We use the def kernel, so a gaussian, which we can do 1D
#sigma = self.kernel_params['sigma']
# this is [(1, 0), (0, 1)], which should be equivalent to a sigma of 1
sigma = np.array([1], dtype = np.float64)

sigma = np.sqrt(sigma) #for generic reasons

pers_range = (0, 1.01)
pixel_size = (pers_range[1] - pers_range[0]) * 0.05
resolution = int((pers_range[1] - pers_range[0]) / pixel_size)
pers_pnts = np.array(np.linspace(pers_range[0], pers_range[1] + pixel_size,
                                resolution, endpoint=False, dtype=np.float64))

for i in range(n):
#    ncdf_b = kernels._norm_cdf((self._bpnts - pers_dgm[i, 0]) / sigma)
# Birth doesn't matter for this
#    ncdf_p = kernels._norm_cdf((self._ppnts - pers_dgm[i, 1]) / sigma)
# _ppnts = np.array(np.linspace(self._pers_range[0], self._pers_range[1] + self._pixel_size,
#                                self._resolution[1] + 1, endpoint=False, dtype=np.float64))
    ncdf_persistence = PersistenceImages.cdfs._norm_cdf((pers_pnts - pers_dgm[i, 1]) / sigma)
    pers_img += wts[i]*ncdf_persistence

persim.plot_diagrams(pers_dgm)

# create a figure and axis
fig, ax = plt.subplots()

# plot the persistence image as a line
ax.imshow(pers_img.reshape((1, -1)).T,  origin='lower', vmin=0, vmax=10)
#ax.invert_yaxis()
# Show the plot
plt.show()

pers_img.reshape((1, -1)).T






