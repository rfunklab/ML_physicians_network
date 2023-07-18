# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:58:49 2022

CMD line CV script
Run cross validation using a given algorithm to generate scores for the parameters defined in cv_prep_vars

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
# algorithm_to_use = 'rfr'
# year = '2011'
# outcome_to_use = 'ar_stdprice_total'
# pixel_resolution = cv_prep_vars.pixel_resolution_map['mid']; pixel_opt = 'mid'

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

#%% Collect input

# Help message
def help():
    print("""Run cross validation using a given algorithm to generate scores for the parameters defined in cv_prep_vars
          
-a, --algorithm    Algorithm to fit. rfr: Random Forest Regression, lasso, and en: Elastic Net suported
-y, --year         YYYY format for available years
-o, --outcome      The name of a column from the outcome data
-r, --resolution   small (0.025), mid (0.05), and large (0.075) supported
-d, --dim          Integer from 0 on for the H dimension to run
-m, --metric       Metric to use for scoring. mse (Mean Squared Error, default), mae (Mean Absolute Error), and r2 (R^2) supported
-k, --kfolds       Number of folds. Used to find the json file that should have been generated by generate_kfolds_indices
-t, --test_prc     Percent of data used for testing. Used to find the json file that should have been generated by generate_kfolds_indices
-s, --seed         Used to provide a seed or generate a seed (if True). Otherwise, the stored seed will be used (52454). Integer or bool.
-w, --overwrite    Used to overwrite previously stored data. Default is False, and so will try to continue where it left off.
-h, --help         Print this message and exit.
""")

required_opts = [('-a', '--algorithm'), ('-y', '--year'), ('-o', '--outcome'), ('-r', '--resolution')]
options, remainder = getopt.getopt(sys.argv[1:], "a:y:o:r:d:m:k:t:s:w:h", 
                                   ["algorithm=","year=","outcome=","resolution=","dim=","metric=","kfolds=","test_prc=", "seed=", "overwrite=", "help"])

for opt, arg in options:
    # Check required opts
    if len(required_opts) != 0: #This means that they have all ben matched
        result = [opt in l for l in required_opts]
        
        if True in result: #A requred opt was found
            index_opt = result.index(True)
            _ = required_opts.pop(index_opt)
    
    if opt in ('-a', '--algorithm'):
        if arg.lower() in ['rfr', 'lasso', 'en']:
            algorithm_to_use = arg.lower()
        else:
            sys.exit("Only algorithms available are: rfr, lasso, en")
    elif opt in ('-y', '--year'):
        year = str(arg)
    elif opt in ('-o', '--outcome'):
        outcome_to_use = arg
    elif opt in ('-r', '--resolution'):
        if arg.lower() in ['small', 'mid', 'large']:
            pixel_opt        = arg.lower()
            pixel_resolution = cv_prep_vars.pixel_resolution_map[pixel_opt]
        elif cv_prep_vars.is_float(arg) and (float(arg) > 0 and float(arg) < 1):
            pixel_opt        = arg
            pixel_resolution = float(arg)
        else:
            sys.exit("Please select small, mid, large or a number greater than 0 and less than 1.")
    elif opt in ('-d', '--dim'):
        if arg.isnumeric():
            hdim = arg
        else:
            sys.exit("hdim must be an integer")
    elif opt in ('-m', '--metric'):
        if arg.lower() in ['mse', 'mae', 'r2']:
            metric_str = arg.lower()
        else:
            sys.exit("Only metrics available are: mse, mea, r2. See help for definitions.")
    elif opt in ('-k', '--kfolds'):
        k = int(arg)
    elif opt in ('-t', '--test_percent'):
        test_size_percent = float(arg)
    elif opt in ('-s', '--seed'):
        if arg.isnumeric():
            seed_opt = int(arg)
        elif arg.lower() in ['t', 'true', 'f', 'false']:
            seed_opt = bool(arg)
    elif opt in ('-w', '--overwrite') and opt.lower() in ['t', 'true', 'f', 'false']:
        overwrite_opt = bool(arg)
    elif opt in ('-h', '--help'):
        help()
        sys.exit()

# Exit if there are missing required opts
if len(required_opts) != 0:
    help()
    sys.exit("The following options are missing: " + str(required_opts))



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
                    "_res-{resolution}_hdim-{hdim}_metric-{metric}_seed-{seed}_v-3_desc-"

script_name = savename_template.format_map(opts_info)

output_fn          = op.join(cv_prep_vars.CV_OUTPUT, script_name + 'cv_results.txt')
last_save_state_fn = op.join(cv_prep_vars.CV_OUTPUT, script_name + 'last_save_state.pickle')


#%% Outcome
# Load outcome data
outcome_df = cv_prep_vars.get_outcome_df(outcome_to_use)

# Extract outcome data (outcome, year, and HSA) and drop NA
relevant_outcome_data = outcome_df[['hsa', 'year', outcome_to_use]]
relevant_outcome_data = relevant_outcome_data[relevant_outcome_data.year == int(year)]
relevant_outcome_data = relevant_outcome_data.dropna()


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


#%% Make sure we have enough data
hsa_with_data = 0
for index in train_index:
    hsa = data_pds.loc[index]['hsa']
    if int(hsa) in relevant_outcome_data['hsa'].tolist():
        hsa_with_data += 1

if hsa_with_data < np.floor(len(train_index)/2):
    print("Not enough training data.")
    sys.exit(1)  # Exit the script with a non-zero status code indicating an error


test_index = cv_prep_dict['test_index']
hsa_with_data = 0
for index in test_index:
    hsa = data_pds.loc[index]['hsa']
    if int(hsa) in relevant_outcome_data['hsa'].tolist():
        hsa_with_data += 1

if hsa_with_data < np.floor(len(test_index)/2):
    print("Not enough test data.")
    sys.exit(1)  # Exit the script with a non-zero status code indicating an errors
    

#%% Load Save State or Set Seed
# initialize this bool for later use (makes sure that a pers_imgr is generated if starting from a different row)
reinit_pers_imgr = False

# initialize to being at the start value
start_val = 50000
start_row = start_val

if op.exists(last_save_state_fn) and not overwrite_opt:
    # Load previous save data
    last_cv_data     = pd.read_csv(output_fn, dtype = str)
    last_cv_data.alg = last_cv_data.alg.astype(int)
    last_row_saved   = last_cv_data.shape[0] #starts with 1
    
    if last_row_saved == 0:
        start_row = start_val
    else:
        start_row = last_row_saved + start_val
        
        ## Update parameter df
        # append previous data to param DF
        concat_df = pd.concat([param_df, last_cv_data])
        
        # remove duplicated indices (keeping appended date) and then sort by index
        param_df = concat_df[~concat_df.index.duplicated(keep = 'last')].sort_index()
        
        # Make sure to re-init pers_imgr
        reinit_pers_imgr = True
        
        ## Load save state
        with open(last_save_state_fn, 'rb') as file:
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

for row in param_df.index[start_row:75000]:
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
    if (row == 0) or (row == start_val) or (wgt_fx != str(param_df.loc[row-1, 'wgt_fx'])) or (wgt_fx_theta != str(param_df.loc[row-1, 'wgt_fx_theta'])) or reinit_pers_imgr:
        pers_imgr = cv_prep_vars.get_pers_imgr(wgt_fx, wgt_fx_theta, cv_prep_vars.wgt_fx_dict, pixel_resolution_info['pers_max'], pixel_resolution_info)
        
        # turn off trigger to re-init pers_imgr
        reinit_pers_imgr = False

        #%% Generate PIs for new imgr
        curr_img_param_data = []
        unavailable_indices = [] # indices that are NA
        avail_indices       = []
        for index in train_index:
            # If there is enough data, it still might be NA and have been dropped
            # Make sure that the HSA is in the releveant outcome data before continuing
            curr_hsa = data_pds.loc[index]['hsa']
            if int(curr_hsa) not in relevant_outcome_data['hsa'].tolist():
                unavailable_indices.append(index)
                continue
            else:
                avail_indices.append(index)
            
            curr_hsa_data = cv_prep_vars.get_PI_and_outcome(data = data_pds.loc[index],
                                                            year = year,
                                                            hdim = hdim,
                                                            outcome_data = relevant_outcome_data,
                                                            outcome_col = outcome_to_use,
                                                            vec_len = vec_len,
                                                            pers_imgr = pers_imgr)
            curr_img_param_data.append(curr_hsa_data)

        curr_img_param_data       = pd.DataFrame(curr_img_param_data)
        curr_img_param_data.index = avail_indices

        #%% Prep data for each fold
        # For each fold, collect the training/validation X/y data in a dict
        kf_data = {n:[] for n in range(k)}
        for fold in range(k):
            # get indices for train/validation for the current fold
            curr_kf_train_index      = [train_index[sub_index] for sub_index in kf_train_folds[str(fold)]]
            curr_kf_validation_index = [train_index[sub_index] for sub_index in kf_validation_folds[str(fold)]]
            
            # Remove NA data
            curr_kf_train_index      = [index for index in curr_kf_train_index if index not in unavailable_indices]
            curr_kf_validation_index = [index for index in curr_kf_validation_index if index not in unavailable_indices]

            # use indices to get train/validation data for the current fold
            curr_kf_train_data      = curr_img_param_data.loc[curr_kf_train_index]
            curr_kf_validation_data = curr_img_param_data.loc[curr_kf_validation_index]

            # split the train/validation data for the current fold into X/y
            curr_kf_train_X = curr_kf_train_data[hdim_cols]
            curr_kf_train_y = curr_kf_train_data[outcome_col_y]
            
            curr_kf_validation_X = curr_kf_validation_data[hdim_cols]
            curr_kf_validation_y = curr_kf_validation_data[outcome_col_y]

            # combine all current fold data into a list [train_X, train_y, validation_X, validation_y]
            kf_data[fold] = [curr_kf_train_X, curr_kf_train_y, curr_kf_validation_X, curr_kf_validation_y]
                                
    #%% Fit data to alg
    # Get classifier
    classifier = cv_prep_vars.get_ml_alg(alg, ml_theta1, **other_ml_thetas)

    # Loop folds
    for fold in range(k):
        curr_fold_data = kf_data[fold]

        curr_classifier_train_X, curr_classifier_train_y, curr_classifier_validation_X, curr_classifier_validation_y = curr_fold_data
        
        # Preproc train data
        train_X_scaler, train_y_scaler = StandardScaler(), StandardScaler()
        
        curr_classifier_train_X = train_X_scaler.fit_transform(curr_classifier_train_X)
        curr_classifier_train_y = train_y_scaler.fit_transform(np.array(curr_classifier_train_y).reshape(-1, 1))

        # Preproc test data
        curr_classifier_validation_X = train_X_scaler.transform(curr_classifier_validation_X)
        curr_classifier_validation_y = train_y_scaler.transform(np.array(curr_classifier_validation_y).reshape(-1, 1))
        
        # Fit data
        classifier.fit(curr_classifier_train_X, np.ravel(curr_classifier_train_y))

        # Append score
        curr_classifier_score = metric_to_use(curr_classifier_validation_y, classifier.predict(curr_classifier_validation_X))

        param_df.loc[row, str(fold)] = curr_classifier_score


    #%% Save data every 1500 iterations
    if (row - start_row) % 1500 == 0:
        # Save state
        last_save_state = np.random.get_state()
        with open(last_save_state_fn, 'wb') as file:
            pickle.dump(last_save_state, file)
        
        # Save output
        ##Keep only rows where the final column of results have values
        data_to_save = param_df[~param_df['4'].isna()]
        data_to_save.to_csv(output_fn, index=False)


#%% Clean up
# If the script reaches here, then it has completed and should save final version
data_to_save = param_df[~param_df['4'].isna()]
data_to_save.to_csv(output_fn, index=False)

# and then delete some files
os.remove(last_save_state_fn)


