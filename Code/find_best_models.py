#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:58:49 2022


@author: fpichard
"""

#%% Modules
import sys
import os
import getopt
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IN-HOUSE IMPORTS
sys.path.append('./Code/modules')
import cv_prep_vars


#%% FUNCTIONS
def describe_helper(series):
    """ Cleans up a pandas describe series
    
    src: https://stackoverflow.com/a/61887592
    """
    splits = str(series.describe()).split()
    keys, values = "", ""
    for i in range(0, len(splits), 2):
        keys += "{:8}\n".format(splits[i])
        values += "{:>8}\n".format(splits[i+1])
    return keys, values

def hist_with_summary_stats(df, xlabel = ""):
    """
    src: https://stackoverflow.com/a/61887592
    """
    
    plt.hist(df, bins=10)
    plt.figtext(.95, .49, describe_helper(pd.Series(df))[0], {'multialignment':'left'})
    plt.figtext(1.05, .49, describe_helper(pd.Series(df))[1], {'multialignment':'right'})
    plt.ylabel('Count')
    plt.xticks(rotation = 25)
    
    # Submitted args
    plt.xlabel(xlabel)
    plt.title("Histogram of " + xlabel)
    plt.show()

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
year             = '2011'
# Select outcome column name to use
outcome_to_use   = 'ar_stdprice_total'
# Select pixel resolution for PIs: small, mid, large
pixel_resolution = 'mid'

##
# Opt Args
#%% Set Defaults
##
# Decide on which dim to use
hdim             = '0'
seed_opt         = 1407212683


#%% Misc variables
algorithms_to_use = ['rfr', 'lasso', 'en']

pd_describe_keys = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

avg_score_column = 'mean_score'

#%% Collect input

# Help message
def help():
    print("""Select the best models
          
-y, --year         YYYY format for available years
-o, --outcome      The name of a column from the outcome data
-r, --resolution   small (0.025), mid (0.05), and large (0.075) supported
-d, --dim          Integer from 0 on for the H dimension to run
-s, --seed         Provide the seed used for generating this data. Integer.
-h, --help         Print this message and exit.
""")

required_opts = [('-y', '--year'), ('-o', '--outcome'), ('-r', '--resolution')]
options, remainder = getopt.getopt(sys.argv[1:], "y:o:r:d:s:h", 
                                   ["year=","outcome=","resolution=","dim=","seed=","help"])

for opt, arg in options:
    # Check required opts
    if len(required_opts) != 0: #This means that they have all ben matched
        result = [opt in l for l in required_opts]
        
        if True in result: #A requred opt was found
            index_opt = result.index(True)
            _ = required_opts.pop(index_opt)
    
    elif opt in ('-y', '--year'):
        year = str(arg)
    elif opt in ('-o', '--outcome'):
        outcome_to_use = arg
    elif opt in ('-r', '--resolution'):
        if arg.lower() in ['small', 'mid', 'large']:
            pixel_resolution = arg.lower()
        else:
            sys.exit("Only resolutions available are: small, mid, large")
    elif opt in ('-d', '--dim'):
        if arg.isnumeric():
            hdim = arg
        else:
            sys.exit("hdim must be an integer")
    elif opt in ('-s', '--seed'):
        if arg.isnumeric():
            seed_opt = int(arg)
        else:
            sys.exit("Seed must be an integer. Provided: " + str(seed_opt))
    elif opt in ('-h', '--help'):
        help()
        sys.exit()

# Exit if there are missing required opts
if len(required_opts) != 0:
    help()
    sys.exit("The following options are missing: " + str(required_opts))



#%% Filenames
info_for_run = {'outcome': outcome_to_use,
                'year': year,
                'resolution': pixel_resolution,
                'hdim': hdim,
                'seed': str(seed_opt)}

general_filename_template = 'cv_alg-{{algorithm}}_outcome-{outcome}_year-{year}_res-{resolution}_hdim-{hdim}_seed-{seed}_desc-'
# general_filename_template = 'cv_alg-{{algorithm}}_outcome-{outcome}_res-{resolution}_hdim-{hdim}_seed-{seed}_desc-'
curr_filename_template    = general_filename_template.format_map(info_for_run)


#%% Load CV results
cv_results_all  = []
for algorithm in algorithms_to_use:
    curr_filename = op.join(cv_prep_vars.CV_OUTPUT, curr_filename_template.format_map({'algorithm': algorithm}) + 'cv_results.txt')
    results_df    = pd.read_csv(curr_filename)
    
    cv_results_all.append(results_df)

#TODO
# Convert all parameter vales to str so that the concat doesn't force any to na

# Combine all into one DF
cv_results_all = pd.concat(cv_results_all).reset_index()


#%% Average and sort results
cv_results_all[avg_score_column]  = cv_results_all[[str(num) for num in range(5)]].mean(axis = 1)
cv_results_all.sort_values(by = avg_score_column, inplace=True)


#%% EXPLORE AVERAGE SCORES
avg_score_summary = cv_results_all[avg_score_column].describe()

# Extract data
## Lower 25%
cv_results_lower_25p = cv_results_all.loc[cv_results_all[avg_score_column] <= avg_score_summary["25%"]]

## Lower 50%
cv_results_lower_50p = cv_results_all.loc[cv_results_all[avg_score_column] <= avg_score_summary["50%"]]


#%% Plots
# Show hist for all
hist_with_summary_stats(cv_results_all[avg_score_column], "Mean Score")

# Show hist for lower 25%
hist_with_summary_stats(cv_results_lower_25p[avg_score_column], "Mean Score of Lower 25%")

# Show hist for lower 50%
hist_with_summary_stats(cv_results_lower_50p[avg_score_column], "Mean Score of Lower 50%")


#%% Get percentage of parameters
# Lower 25%
## Percent for algorithms
cv_results_lower_25p.alg.value_counts()/cv_results_lower_25p.alg.value_counts().sum()

## Percent for PI weight function
cv_results_lower_25p.wgt_fx.value_counts()/cv_results_lower_25p.wgt_fx.value_counts().sum()

# Lower 50%
## Percent for algorithms
cv_results_lower_50p.alg.value_counts()/cv_results_lower_50p.alg.value_counts().sum()

## Percent for PI weight function
cv_results_lower_50p.wgt_fx.value_counts()/cv_results_lower_50p.wgt_fx.value_counts().sum()

"""
This doesn't matter because EN has way, way more models than the others, and RFR has way more than Lasso
"""


#%% Algorithm Checks

cv_results_dict = {}
for algorithm in ['rfr', 'lasso', 'en']:
    alg_num = cv_prep_vars.alg_to_num[algorithm]
    
    # Extract results
    curr_cv_results            = cv_results_all.loc[cv_results_all.alg == alg_num]
    cv_results_dict[algorithm] = curr_cv_results
    
    # Hist
    hist_with_summary_stats(curr_cv_results[avg_score_column], "Mean Score of " + algorithm)

rfr_df = cv_results_dict['rfr']
rfr_df.mean_score.describe()

"""
RFR has the lowest values
    count    42601.000000
    mean         0.004469
    std          0.000027
    min          0.004376
        Lower than EN min
    25%          0.004455
    50%          0.004475
    75%          0.004480
    max          0.004572
        Lower than Lasso min
        Lower than EN 25%

All values are lower than all Lasso scores and lower than at least 75% of EN scores.
    What does EN's best 25% look like?
"""

#%% EN's best 25%
en_data              = cv_results_dict['en']
avg_en_score_summary = en_data[avg_score_column].describe()

# Extract data
## Lower 25%
en_cv_results_best_25p = en_data.loc[en_data[avg_score_column] <= avg_en_score_summary["25%"]]

## Plot
hist_with_summary_stats(en_cv_results_best_25p[avg_score_column], "Mean Score of EN's 25% Best")

en_cv_results_best_25p[avg_score_column].describe()

"""
count    94338.000000
mean         0.004651
std          0.000125
min          0.004453
25%          0.004542
50%          0.004639
75%          0.004745
max          0.004913

There are more EN 25% best models than RFR models overall.
They overlap from RFR 25% to below EN 50% best.

I'll throw out the lasso data and focus on all RFR and the 25% best EN.

Questions:
    What can we learn about the sparsity of the best models?
        Focus on EN
    What can we learn about the wgt fx and the best parameters for that?
        Is there a best wgt fx?
            This is hard to answer because there are way more ramp parameters!
            Also those are 2D parameters.
        What are the better values for each?
            How does it vary by algorithm.
    What is the issue with the saturated scores for lasso and EN?
        Is this the best metric?
"""

#%% Sparsity in EN
# the second theta is related to the ratio of L1/L2
#   0: no sparsity, 1: max sparsity



#%% Reminders
#{'rfr': 0, 'lasso': 1, 'en': 2}
#{0: {'label': 'persistence'}, 1: {'label': 'linear_ramp'}}