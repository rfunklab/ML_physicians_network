#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:19:02 2022

@author: fpichard
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

#IN-HOUSE IMPORTS
sys.path.append('./Code/modules')
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
pixel_resolution = ''

# #TESTING
algorithm_to_use = 'lasso'
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
data_pds_fn_template = "proj-PI_year-{year}_region-great_lakes_desc-PD_as_str.csv"

# To load KFolds indices
cv_prep_fn_template = "proj-PI_year-{year}_region-great_lakes_k-{k}_" + \
                    "test-{test_prc}_seed-{seed}_desc-cv_prep_data.json"

# To save output from this script
savename_template = "cv_alg-{algorithm}_outcome-{outcome}_year-{year}_region-great_lakes" +\
                    "_res-{resolution}_hdim-{hdim}_seed-{seed}_desc-"

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


#%% Alg parameters to use
# Load algorithm thetas
all_thetas  = cv_prep_vars.param_thetas_dict[algorithm_to_use]

# Combine all thetas into a dataframe
alg_num   = cv_prep_vars.alg_to_num[algorithm_to_use]
param_df  = cv_prep_vars.get_alg_param_df(all_thetas, alg_num)

# Extract other useful info
num_theta = cv_prep_vars.alg_dict[alg_num]['theta_num']
norm_data = cv_prep_vars.alg_dict[alg_num]['norm']

# Fix RFR issue
if algorithm_to_use == 'rfr':
    param_df.ml_theta2 = param_df.ml_theta2.astype(int) #needed to use as key in dict
    param_df.ml_theta3 = param_df.ml_theta3.astype(int) #need to be int or 0-1 float

param_df = param_df.join(pd.DataFrame(columns = [str(n) for n in range(k)]))


#%% Prep PI vars
pixel_resolution_info = cv_prep_vars.get_resolution_info(pixel_resolution)

pixel_resolution_val = pixel_resolution_info['resolution']
img_shape            = pixel_resolution_info['img_shape']
vec_len              = pixel_resolution_info['vec_len']

# These columns are used to extract the PI columns from the data.
# They are always in the location of the h0 columns.
hdim_cols = pixel_resolution_info['h0_cols']


#%% CV prep
cv_prep_fn = op.join(cv_prep_vars.DATA_PATH, cv_prep_fn_template.format_map(opts_info))

with open(cv_prep_fn) as handle:
    cv_prep_dict = json.loads(handle.read())

train_index         = cv_prep_dict['train_index']
kf_train_folds      = cv_prep_dict['train_folds']
kf_validation_folds = cv_prep_dict['validation_folds']


#%% Load Save State or Set Seed
# initialize this bool for later use (makes sure that a pers_imgr is generated if starting from a different row)
reinit_pers_imgr = False

# initialize to being at 0
start_row = 0

##
#%%   Run CV
##

row = 0

## Get parameter data
row_data = param_df.loc[row]

wgt_fx       = 0
wgt_fx_theta = 1
alg          = row_data['alg']
ml_theta1    = row_data['ml_theta1']

other_ml_thetas = {}
for n in range(2, num_theta + 1):
    new_key = 'ml_theta' + str(n)
    other_ml_thetas[new_key] = row_data[new_key]


## Prep data
# Create new pers_imgr 
pers_imgr = cv_prep_vars.get_pers_imgr(wgt_fx, wgt_fx_theta, cv_prep_vars.wgt_fx_dict, pixel_resolution_info['pers_max'], pixel_resolution_info)

#%% Generate PIs for new imgr
curr_img_param_data = []
for index in train_index:
    curr_hsa_data = cv_prep_vars.get_PI_and_outcome(data = data_pds.loc[index],
                                                    year = year,
                                                    hdim = hdim,
                                                    outcome_data = outcome_df,
                                                    outcome_col = outcome_to_use,
                                                    vec_len = vec_len,
                                                    pers_imgr = pers_imgr)
    curr_img_param_data.append(curr_hsa_data)

curr_img_param_data       = pd.DataFrame(curr_img_param_data)
curr_img_param_data.index = train_index

#%% Preproc Data
# Preproc train data
X_scaler, y_scaler = StandardScaler(), StandardScaler()

curr_img_param_data[hdim_cols] = X_scaler.fit_transform(curr_img_param_data[hdim_cols])
curr_img_param_data[outcome_col_y] = y_scaler.fit_transform(np.array(curr_img_param_data[outcome_col_y]).reshape(-1, 1))

# %% Save data
curr_img_param_data.to_csv("test_for_en.csv", index = False)
