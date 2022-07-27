#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:30:20 2022

@author: fpichard
"""

#%% Modules
import sys
import getopt
import json
import os.path as op
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

#IN-HOUSE IMPORTS
sys.path.append('./Code/modules/')
import cv_prep_vars


##
#%%   COLLECT INPUT
##

year             = '' #2011
# Number of folds to use
k                = 5
# Percent of sample to use for testing
test_size_pct    = 0.33
# Seed to use or generate
seed_opt         = 52454


#%% Collect input

required_opts = [('-y', '--year')]
options, remainder = getopt.getopt(sys.argv[1:], "y:k:t:s:", ["year=","kfolds=","test_percent=", "seed="])

for opt, arg in options:
    # Check required opts
    if len(required_opts) != 0: #This means that they have all ben matched
        result = [opt in l for l in required_opts]
        
        if True in result: #A requred opt was found
            idx_opt = result.index(True)
            _ = required_opts.pop(idx_opt)
    
    if opt in ('-y', '--year'):
        year = str(arg)
    elif opt in ('-k', '--kfolds'):
        k = int(arg)
    elif opt in ('-t', '--test_percent'):
        test_size_pct = float(arg)
    elif opt in ('-s', '--seed'):
        if opt.isnumeric():
            seed_opt = int(opt)
        elif opt.lower() in ['t', 'true', 'f', 'false']:
            seed_opt = bool(opt)

# Exit if there are missing required opts
if len(required_opts) != 0:
    sys.exit("The following options are missing: " + str(required_opts))


#%% Load PD data
data_pds_fn = op.join(cv_prep_vars.DATA_PATH, "proj-PI_year-" + year + "_region-great lakes_desc-PD as str.csv")
data_pds    = pd.read_csv(data_pds_fn)

# change nas to '' to help with the conversion
data_pds.fillna('', inplace = True)


#%% SET SEED
if type(seed_opt) == int:
    seed_generated = seed_opt
else:
    seed_generated = np.random.randint(2**32)


#%% CV Prep
# train/test split: hsa ids
hsa_train, hsa_test = train_test_split(data_pds.hsa, test_size = test_size_pct)
train_idx, test_idx = hsa_train.index.tolist(), hsa_test.index.tolist()

# cv split: k-fold
kf     = KFold(n_splits=k, shuffle=True, random_state=seed_generated)
kf_idx = list(kf.split(train_idx))

kf_train_folds      = {fold:elem.tolist() for fold, elem in enumerate([elem[0] for elem in kf_idx])}
kf_validation_folds = {fold:elem.tolist() for fold, elem in enumerate([elem[1] for elem in kf_idx])}


#%% Save CV Prep Data
# Prep to save
cv_data = {
    'train_idx'       : train_idx,
    'test_idx'        : test_idx,
    'train_folds'     : kf_train_folds,
    'validation_folds': kf_validation_folds
    }

# Save
savefn = op.join(cv_prep_vars.DATA_PATH, "proj-PI_year-" + year + "_region-great lakes_k-" + str(k) + "_test-" + str(test_size_pct) + "_desc-cv_prep_data.json")

with open(savefn, 'w') as fp:
    json.dump(cv_data, fp)