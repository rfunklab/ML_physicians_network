# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:58:49 2022

CMD line CV script

@author: fpichard
"""

#%% Modules
import sys
import getopt
import json
import pickle
import os.path as op
import numpy as np
import pandas as pd
from sklearn import metrics

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
alg_to_use       = ''
year             = '' #2011
# Select outcome column name to use
outcome_to_use   = '' #ar_stdprice_total
# Select pixel resolution for PIs: small, mid, large
pixel_resolution = ''

# #TESTING
# alg_to_use = 'lasso'
# year = '2011'
# outcome_to_use = 'ar_stdprice_total'
# pixel_resolution = 'mid'

##
# Opt Args
#%% Set Defaults
##
# Decide on which dim to use
hdim             = '0'
# Select Metric for Scoring:
metric_to_use    = metrics.mean_squared_error
# Number of folds to use
k                = 5
# Percent of sample to use for testing
test_size_pct    = 0.33
# Seed to use or generate
seed_opt         = 52454
# Overwrite previous saved info (will not try to start from where it left off)
overwrite_opt    = False

# Misc vars
outcome_col_y = 1

#%% Collect input

required_opts = [('-a', '--algorithm'), ('-y', '--year'), ('-o', '--outcome'), ('-r', '--resolution')]
options, remainder = getopt.getopt(sys.argv[1:], "a:y:o:r:h:m:k:t:s:w:", ["algorithm=","year=","outcome=","resolution=","hdim=","metric=","kfolds=","test_percent=", "seed=", "overwrite="])

for opt, arg in options:
    # Check required opts
    if len(required_opts) != 0: #This means that they have all ben matched
        result = [opt in l for l in required_opts]
        
        if True in result: #A requred opt was found
            idx_opt = result.index(True)
            _ = required_opts.pop(idx_opt)
    
    if opt in ('-a', '--algorithm'):
        if arg.lower() in ['rfr', 'lasso', 'en']:
            alg_to_use = arg.lower()
        else:
            sys.exit("Only algorithms available are: rfr, lasso, en")
    elif opt in ('-y', '--year'):
        year = str(arg)
    elif opt in ('-o', '--outcome'):
        outcome_to_use = arg
    elif opt in ('-r', '--resolution'):
        if arg.lower() in ['small', 'mid', 'large']:
            pixel_resolution = arg.lower()
        else:
            sys.exit("Only resolutions available are: small, mid, large")
    elif opt in ('-h', '--hdim'):
        if arg.isnumeric():
            hdim = arg
        else:
            sys.exit("hdim must be an integer")
    elif opt in ('-m', '--metric'):
        print('Currently only using mean squared error.')
    elif opt in ('-k', '--kfolds'):
        k = int(arg)
    elif opt in ('-t', '--test_percent'):
        test_size_pct = float(arg)
    elif opt in ('-s', '--seed'):
        if arg.isnumeric():
            seed_opt = int(arg)
        elif arg.lower() in ['t', 'true', 'f', 'false']:
            seed_opt = bool(arg)
    elif opt in ('-w', '--overwrite') and opt.lower() in ['t', 'true', 'f', 'false']:
        overwrite_opt = bool(arg)

# Exit if there are missing required opts
if len(required_opts) != 0:
    sys.exit("The following options are missing: " + str(required_opts))


#%% Generate Seed
if type(seed_opt) == int:
    seed_generated = seed_opt
else:
    seed_generated = np.random.randint(2**32)


#%% Save names
script_name       = 'cv_alg-' + alg_to_use + '_outcome-' + outcome_to_use + '_res-' + pixel_resolution + '_hdim-' + hdim + '_seed-' + str(seed_generated) + '_desc-' 
output_fn         = op.join(cv_prep_vars.CV_OUTPUT, script_name + 'cv_results.txt')
last_save_stat_fn = op.join(cv_prep_vars.CV_OUTPUT, script_name + 'last_save_state.pickle')


#%% Outcome
# Load outcome data
outcome_df = cv_prep_vars.get_outcome_df(outcome_to_use)


#%% Load PD data
data_pds_fn = op.join(cv_prep_vars.DATA_PATH, "proj-PI_year-" + year + "_region-great lakes_desc-PD as str.csv")
data_pds    = pd.read_csv(data_pds_fn)

# change nas to '' to help with the conversion
data_pds.fillna('', inplace = True)


#%% Alg parameters to use
param_df  = cv_prep_vars.param_df_dict[alg_to_use]
num_theta = cv_prep_vars.alg_dict[cv_prep_vars.alg_to_num[alg_to_use]]['theta_num']
norm_data = cv_prep_vars.alg_dict[cv_prep_vars.alg_to_num[alg_to_use]]['norm']

# prepoc func
if norm_data:
    data_preproc = cv_prep_vars.better_normalize
else:
    # do nothing
    data_preproc = lambda x: x

param_df = param_df.join(pd.DataFrame(columns = [str(n) for n in range(k)]))


#%% Prep PI vars
pixel_resolution_info = cv_prep_vars.pixel_resolution_info[pixel_resolution]

pixel_resolution_val = pixel_resolution_info['resolution']
img_shape            = pixel_resolution_info['img_shape']
vec_len              = pixel_resolution_info['vec_len']

# These columns will always include the PI for the first or only generated HDim
hdim_cols = pixel_resolution_info['h0_cols']


#%% CV prep
cv_prep_fn = op.join(cv_prep_vars.DATA_PATH, "proj-PI_year-" + year + "_region-great lakes_k-" + str(k) + "_test-" + str(test_size_pct) + "_desc-cv_prep_data.json")
with open(cv_prep_fn) as handle:
    cv_prep_dict = json.loads(handle.read())

train_idx           = cv_prep_dict['train_idx']
kf_train_folds      = cv_prep_dict['train_folds']
kf_validation_folds = cv_prep_dict['validation_folds']


#%% Load Save State or Set Seed
# initialize this bool for later use (makes sure that a pers_imgr is generated if starting from a different row)
reinit_pers_imgr = False

# initialize to being at 0
start_row = 0

if op.exists(last_save_stat_fn) and not overwrite_opt:
    # Load previous save data
    last_cv_data     = pd.read_csv(output_fn, dtype = str)
    last_cv_data.alg = last_cv_data.alg.astype(int)
    last_row_saved   = last_cv_data.shape[0] #starts with 1
    
    if last_row_saved == 0:
        start_row = 0
    else:
        start_row = last_row_saved
        
        ## Update parameter df
        # append previous data to param DF
        concat_df = pd.concat([param_df, last_cv_data])
        
        # remove duplicated indices (keeping appended date) and then sort by index
        param_df = concat_df[~concat_df.index.duplicated(keep = 'last')].sort_index()
        
        # Make sure to re-init pers_imgr
        reinit_pers_imgr = True
        
        ## Load save state
        with open(last_save_stat_fn, 'rb') as file:
            last_save_state = pickle.load(file)
        np.random.set_state(last_save_state)
else:
    # prep output file
    pd.DataFrame(param_df.columns.values).T.to_csv(output_fn, header = False, index = False)
    
    # Set seed
    np.random.seed(seed_generated)


##
#%%   Run CV
##

for row in param_df.index[start_row:]:
    ## Get parameter data
    row_data = param_df.loc[row]

    wgt_fx       = row_data['wgt_fx']
    wgt_fx_theta = row_data['wgt_fx_theta']
    alg          = row_data['alg']
    ml_theta1    = row_data['ml_theta1']

    other_ml_thetas = {}
    for n in range(2, num_theta + 1):
        new_key = 'ml_theta' + str(n)
        other_ml_thetas[new_key] = row_data[new_key]


    ## Prep data
    # Create new pers_imgr if needed and prep KF data dict
    if (row == 0) or (wgt_fx != str(param_df.loc[row-1, 'wgt_fx'])) or (wgt_fx_theta != str(param_df.loc[row-1, 'wgt_fx_theta'])) or reinit_pers_imgr:
        pers_imgr = cv_prep_vars.get_pers_imgr(wgt_fx, wgt_fx_theta, cv_prep_vars.wgt_fx_dict, pixel_resolution_info['pers_max'], pixel_resolution_info)
        
        # turn off trigger to re-init pers_imgr
        reinit_pers_imgr = False

        #%% Generate PIs for new imgr
        curr_img_param_data = []
        for idx in train_idx:
            curr_hsa_data = cv_prep_vars.get_PI_and_outcome(data = data_pds.loc[idx],
                                                            year = year,
                                                            hdim = hdim,
                                                            outcome_data = outcome_df,
                                                            outcome_col = outcome_to_use,
                                                            vec_len = vec_len,
                                                            pers_imgr = pers_imgr)
            curr_img_param_data.append(curr_hsa_data)

        curr_img_param_data       = pd.DataFrame(curr_img_param_data)
        curr_img_param_data.index = train_idx

        #%% Prep data for each fold
        # For each fold, collect the training/validation X/y data in a dict
        kf_data = {n:[] for n in range(k)}
        for fold in range(k):
            # get indices for train/validation for the current fold
            curr_kf_train_idx      = [train_idx[sub_idx] for sub_idx in kf_train_folds[str(fold)]]
            curr_kf_validation_idx = [train_idx[sub_idx] for sub_idx in kf_validation_folds[str(fold)]]

            # use indices to get train/validation data for the current fold
            curr_kf_train_data      = curr_img_param_data.loc[curr_kf_train_idx]
            curr_kf_validation_data = curr_img_param_data.loc[curr_kf_validation_idx]

            # split the train/validation data for the current fold into X/y
            curr_kf_train_X = curr_kf_train_data[hdim_cols]
            curr_kf_train_y = curr_kf_train_data[outcome_col_y]
            
            curr_kf_validation_X = curr_kf_validation_data[hdim_cols]
            curr_kf_validation_y = curr_kf_validation_data[outcome_col_y]

            # combine all current fold data into a list [train_X, train_y, validation_X, validation_y]
            kf_data[fold] = [curr_kf_train_X, curr_kf_train_y, curr_kf_validation_X, curr_kf_validation_y]
                                
    #%% Fit data to alg
    # Get classifier
    clf = cv_prep_vars.get_ml_alg(alg, ml_theta1, **other_ml_thetas)

    # Loop folds
    for fold in range(k):
        curr_fold_data = kf_data[fold]

        curr_clf_train_X, curr_clf_train_y, curr_clf_validation_X, curr_clf_validation_y = [data_preproc(l) for l in curr_fold_data]

        # Fit data
        clf.fit(curr_clf_train_X, curr_clf_train_y)

        # Append score
        curr_clf_score = metric_to_use(curr_clf_validation_y, clf.predict(curr_clf_validation_X))

        param_df.loc[row, str(fold)] = curr_clf_score


    #%% Save data every 50 iterations
    if (row - start_row) % 50 == 0:
        # Save state
        last_save_state = np.random.get_state()
        with open(last_save_stat_fn, 'wb') as file:
            pickle.dump(last_save_state, file)
        
        # Save output
        ##Keep only rows where the final column of results have values
        data_to_save = param_df[~param_df['4'].isna()]
        data_to_save.to_csv(output_fn, index=False)


