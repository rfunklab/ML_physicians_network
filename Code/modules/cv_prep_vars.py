#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:39:39 2022

@author: icd
"""

#%% Modules
import os
import os.path as op
import itertools
import numpy as np
import pandas as pd

import PersistenceImages.persistence_images as pimg
import PersistenceImages.weighting_fxns as wfxs
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import normalize


###
#
#%%   Functions
#
###

def get_outcome_info(hsa_id, year, data, col):
    """
    Pull the outcome data for a HSA ID for a given year

    Parameters
    ----------
    hsa_id : str or int
        HSA ID
    year : str or int
        year to select
    data : dataframe
        dataframe with the outcome data, e.g., loaded cost of care data: referral_network_topology_hsa_stdprices_ffs_wide
    col : str
        Column of data to extract

    Returns
    -------
    int
    """

    return int(data[(data.hsa == int(hsa_id)) & (data.year == int(year))][col])


def get_PI_as_vec_from_str(diagram, vec_len, pers_imgr):
    """
    Calculate the PI and then convert it to a vector from a PD stored as a str

    Parameters
    ----------
    diagram : str
        One dimesion of the results from running a filtration algorithm saved as a str
    vec_len : int
        Expected length of the vector
    pers_img : object

    Returns
    -------
    array
    """

    if diagram == '':
        return np.zeros(vec_len)

    # Convert to array
    diagram = np.array([[float(e2) for e2 in elem.split('-')] for elem in diagram.split(';')])

    # Convert to  PI
    PI = pers_imgr.transform(diagram, skew=True)

    # Convert to vec and return
    return PI.reshape((PI.size))


def get_PI_and_outcome(data, year, hdim, outcome_data, outcome_col, vec_len, pers_imgr):
    """
    Load, pull, and combine outcome and PI information for a given HSA ID (HID) and year

    Parameters
    ----------
    data : Pandas Series
        Row from the stored data with HSA, H0, etc columns. PD information must be stored in str format with - and ; as delimiters
    year : str
        YYYY format year to retrieve outcome info
    hdim : numeric or str
        H-dimension to convert to PIs
    outcome_data: dataframe
        Dataframe with desired outcome data
    outcome_col : str
        Column name for the outcome to select
    vec_len : int
        Expected length of the PI vector
    pers_imgr : obj
        PersistanceImage object for filtration

    Returns
    -------
    array
        HSA, outcome_val, vector values for PI
    """

    # Get outcome
    hsa_outcome = get_outcome_info(data.hsa, year, outcome_data, outcome_col)

    # Get PIs
    hkey = 'h'  + str(hdim)
    if hkey in data.index:
        pi_vec = get_PI_as_vec_from_str(data['h' + str(hdim)], vec_len, pers_imgr)
    else:
        raise ValueError("Data not found for the submitted dimentions: " + str(hdim))

    # Combine all info
    all_info = np.concatenate([[data.hsa], [hsa_outcome], pi_vec])

    return all_info


def get_pers_imgr(wgt_fx_num, wgt_fx_theta_str, wgt_fx_dict, pers_max, resolution_info):
    """
    Generate a persistance imager object based on informatin from the parameter matrix

    Parameters
    ----------
    wgt_fx_num : numeric
        Numeric representation for the given weight function
    wgt_fx_theta_str : str
        Theta value for the given weight function
    wgt_fx_dict : dict
        Dict containing information about the meaning of the weight function column
    pers_max : numeric
        Max persistence value
    resolution_info : dict
        Pixel resolution information to use for PIs, generated from get_resolution_info()

    Returns
    -------
    Persistance Imager object
    """

    wgt_fx_label = wgt_fx_dict[int(wgt_fx_num)]['label']

    if wgt_fx_label == 'persistence':
        wgt_fx              = wfxs.persistence
        wgt_fx_params       = {'n': float(wgt_fx_theta_str)}
    else:
        wgt_fx              = wfxs.linear_ramp
        wgt_fx_theta_list   = [float(elem) for elem in wgt_fx_theta_str.split(',')]
        wgt_fx_params       = { 'low' : 0,
                               'high' : 1, # max weight used in ramp,
                               'start': wgt_fx_theta_list[0], # When to start ramp and set all vals before to 0
                                'end' : wgt_fx_theta_list[1]} # When to stop ramp and set all higher values to max weight

    #One dict to load into the PI obj
    pi_wgt_fx_dict = {       'weight': wgt_fx,
                      'weight_params': wgt_fx_params}

    return pimg.PersistenceImager(pixel_size  = resolution_info['resolution'],
                                  pers_range  = (resolution_info['pers_min'], resolution_info['pers_max']),
                                  birth_range = (resolution_info['birth_min'], resolution_info['birth_max']),
                                       **pi_wgt_fx_dict)


def get_ml_alg(alg, ml_theta1, **kwargs):
    """
    Generate an ML algorithm object based on informatin from the parameter matrix

    # TODO: 'fit_intercept': False??

    Parameters
    ----------
    alg : numeric
        Numeric representation for the given ML algorithm
    ml_theta1 : numeric
        Theta value for the given ML algorithm
    kwargs
        other ml_theta arguments for the algorithms

    Returns
    -------
    SKLearn ML algirthm object
    """

    ml_alg_label = alg_dict[alg]['label']

    if ml_alg_label == 'RFR':
        # Random Forest Regression
        ml_alg      = RFR
        ml_thetas   = {     'n_estimators': int(ml_theta1),
                            'max_features': max_features_dict[int(kwargs['ml_theta2'])],
                       'min_samples_split': kwargs['ml_theta3'],
                               'bootstrap': bool(kwargs['ml_theta4'])}

    elif ml_alg_label == 'Lasso':
        # Lasso
        ml_alg      = Lasso
        ml_thetas   = {        'alpha': float(ml_theta1),
                       'fit_intercept': False,
                            'max_iter': 10000}

    elif ml_alg_label == 'EN':
        # Elastic Net
        ml_alg      = ElasticNet
        ml_thetas   = {        'alpha': float(ml_theta1),
                            'l1_ratio': float(kwargs['ml_theta2']),
                       'fit_intercept': False,
                            'max_iter': 10000}

    return ml_alg(**ml_thetas)


def better_normalize(data):

    if len(np.array(data).shape) < 2:
        return data/np.linalg.norm(data)
    else:
        return normalize(data)


def get_outcome_df(column_name):
    """
    Uses the column name to determine outcome dataset to load

    Parameters
    ----------
    column_name : str
        Column name found in csv data files

    Returns
    -------
    dataframe
    """

    col_name = outcome_data_cols_df.columns[(outcome_data_cols_df == column_name).sum() > 0].values[0]

    if col_name == 'census_data':
        outcome_df = pd.read_csv(outcome_csv_dict[col_name], compression ='gzip')
    else:
        outcome_df = pd.read_csv(outcome_csv_dict[col_name], compression ='gzip')

    return outcome_df


def get_resolution_info(pixel_resolution_size, birth_min = 0, birth_max = 1.01, pers_min = 0, pers_max = 1.01, max_h_dim = 2):
    """
    Uses the pixel resolution to generate required info

    Parameters
    ----------
    pixel_resolution_size : numeric
        Desired resolution
    birth_min : numeric, optional
        defualt is 0
    birth_max : numeric, optional
        defualt is 1.01
    pers_min : numeric, optional
        defualt is 0
    pers_max : numeric, optional
        defualt is 1.01
    max_h_dim : int, optional
        Max H dimension to generate info for. Default is 2.

    Returns
    -------
    dict
    """
    
    # init dict
    pixel_info = {'resolution': pixel_resolution_size,
                  'birth_min': birth_min, 'birth_max': birth_max,
                  'pers_min': pers_min, 'pers_max': pers_max}
    
    birth_rng = birth_max - birth_min
    pers_rng  = pers_max - pers_min
    
    row_size = int(np.floor((1/pixel_resolution_size) * birth_rng))
    col_size = int(np.floor((1/pixel_resolution_size) * pers_rng))
    
    img_shape = (row_size, col_size)
    vec_len   = np.prod(img_shape)   #Length of the vector given the resolution - img shape dims multiplied
    
    pixel_info['img_shape'] = img_shape
    pixel_info['vec_len']   = vec_len
    
    for dim_idx in range(max_h_dim + 1):
        # From one long vector to each H
        rng_h_key             = 'h' + str(dim_idx) + '_vec_rng'
        pixel_info[rng_h_key] = list(range(vec_len * dim_idx, vec_len * (dim_idx + 1)))
        
        # Col idx for h0 etc, starts at 2 [0 is the HSA ID var]
        cols_h_key             = 'h' + str(dim_idx) + '_cols'
        pixel_info[cols_h_key] = list(get_range(vec_len, dim_idx))
    
    return pixel_info


def get_alg_param_df(list_of_all_thetas, alg_num):
    """
    Combine the parameters for the algorithm with the parameters for the weight functions

    Parameters
    ----------
    list_of_all_thetas : list
        List of all list of unique parameter values organized as weight function list, weight funciton theta list, and then algorith thetas lists
    alg_num : int
        Index represeting algorithm

    Returns
    -------
    Dataframe

    """
    
    theta_names = ['ml_theta' + str(idx + 1) for idx in range(len(list_of_all_thetas) - 2)]
    
    ## join all thetas
    all_thetas_mat = np.array(list(itertools.product(*list_of_all_thetas)))
    
    param_df         = pd.DataFrame(all_thetas_mat)
    param_df.columns = ['wgt_fx', 'wgt_fx_theta'] + theta_names
    param_df['alg']  = alg_num
    
    param_df.sort_values(['wgt_fx', 'wgt_fx_theta', 'alg'] + theta_names, inplace = True)
    
    # Keep only (0 and no ',') OR (1 with ',')
    contains_comma_bool = param_df.wgt_fx_theta.str.contains(',')
    wgt_fx_0_bool       = param_df.wgt_fx == '0'
    
    param_df = param_df[(contains_comma_bool & ~wgt_fx_0_bool) | (~contains_comma_bool & wgt_fx_0_bool)]
    param_df.reset_index(inplace=True, drop = True)
    
    return param_df


###
#
#   Config
#
###

#%% Paths
PROJECT_PATH = os.getcwd()
DATA_PATH    = op.join(PROJECT_PATH, "Data")
OUTCOME_DATA = op.join(DATA_PATH, "outcomes")
CV_OUTPUT    = op.join(DATA_PATH, "cv_output")
# REF_NET_PATH = op.join(DATA_PATH, "referral_networks", "files")
# COV_PATH     = op.join(DATA_PATH, "covariates")


#%% Useful files
census_data_csv     = op.join(OUTCOME_DATA, "census_data.csv")
basic_care_csv      = op.join(OUTCOME_DATA, "referral_network_topology_hsa_hedis_6575ffs_wide.csv.gz")
post_acute_care_csv = op.join(OUTCOME_DATA, "referral_network_topology_hsa_postdis_6599ffs_wide.csv.gz")
cost_of_care_csv    = op.join(OUTCOME_DATA, "referral_network_topology_hsa_stdprices_ffs_wide.csv.gz")

outcome_csv_dict = {
        'census_data': census_data_csv,
         'basic_care': basic_care_csv,
    'post_acute_care': post_acute_care_csv,
       'cost_of_care': cost_of_care_csv
    }

outcome_data_cols_df = pd.read_csv(op.join(OUTCOME_DATA, "outcome_data_cols.csv"))

#%% Useful variables
REF_NETWORK_TYPES    = ["hsa", "hrr"]
REF_NETWORK_SUBTYPES = ["local", "extended"]
REF_NETWORK_YEARS    = [str(YR) for YR in list(range(2009, 2018))]
REF_NETWORK_ALGS     = ["hopteaming", "60"]

YEAR_TO_ALG = {"2009": "60",
               "2010": "60",
               "2011": "60",
               "2012": "60",
               "2013": "60",
               "2014": "hopteaming",
               "2015": "hopteaming",
               "2016": "hopteaming",
               "2017": "hopteaming"}

# REF_NETWORK_DIR = op.join(REF_NET_PATH, "{TYPE}", "{SUBTYPE}", "udw", "{YEAR}_{ALG}")
# REF_NETWORK_FILENAME = op.join(REF_NETWORK_DIR, "{ID_NUM}", "{ID_NUM}_{TYPE}_{SUBTYPE}_{YEAR}_{ALG}_G.graphml.gz")



###
#
#%%   Main
#
###

h0_to_h2_weights = [0.75/2, 0.75/2, 0.25]
h0_to_h1_weights = [0.5, 0.5]

seed_generated = 52454

get_range = lambda length, dim: range((length*(0 + dim))+2,(length*(dim + 1)+2))

# Small: 1600, Mid: 400, Large: 169
small_pixel_resolution, mid_pixel_resolution, large_pixel_resolution = 0.025, 0.05, 0.075

#%% Persistence variables
# Def PI parameters
small_pixel_info   = get_resolution_info(small_pixel_resolution)
mid_pixel_info     = get_resolution_info(mid_pixel_resolution)
large_pixel_info   = get_resolution_info(large_pixel_resolution)

# Combine pixel resolution info
pixel_resolution_info = {
    'small': small_pixel_info,
      'mid': mid_pixel_info,
    'large': large_pixel_info
    }

## Wgt fx
wgt_fx_dict = {0: {'label': 'persistence'}, 1: {'label': 'linear_ramp'}}

persistence_thetas       = np.arange(0.5, 3.1, 0.25)
linear_ramp_start_thetas = np.arange(0, 0.3, 0.02)
# A sample of these PDs showed that persistence values were not really higher than 0.33
linear_ramp_end_thetas   = np.concatenate([np.arange(0.14, 0.5, 0.02), [1.01]])
linear_ramp_thetas       = np.array(list(itertools.product(*[linear_ramp_start_thetas, linear_ramp_end_thetas])))
linear_ramp_thetas_str   = np.array([','.join(map(str, elem.tolist())) for elem in linear_ramp_thetas])

wgt_fx_dict[0]['theta_num'] = persistence_thetas.size
wgt_fx_dict[1]['theta_num'] = linear_ramp_thetas_str.size

wgt_fx_thetas    = np.concatenate((persistence_thetas, linear_ramp_thetas_str)).tolist()
wgt_fx_list      = np.concatenate([np.zeros_like(persistence_thetas.astype(int)), np.ones_like(linear_ramp_thetas_str)])
wgt_fx_list_uniq = np.unique(wgt_fx_list).tolist()


#%% ML param matrices

alg_dict   = {0: {'label': 'RFR', 'theta_num': 4, 'norm': True},
              1: {'label': 'Lasso', 'theta_num': 1, 'norm': True},
              2: {'label': 'EN', 'theta_num': 2, 'norm': True}}
alg_to_num = {'rfr': 0, 'lasso': 1, 'en': 2}

max_features_dict = {0: 'auto', 1: 'sqrt', 2: 'log2'}

#this comes from the glmnet package from R
thetas_for_penalty_str = np.exp(np.arange(-9.2, -1.5, 0.09119) + np.random.uniform(0.000001, 0.000002))


#%% RFR
RFR_thetas1 = list(range(100, 400, 50)) #n_estimators
RFR_thetas2 = list(range(3)) #max_features - needs to be an int
RFR_thetas3 = list(range(2, 9, 2)) #min_samples_split
RFR_thetas4 = list(range(2)) #bootstrap - apply bool to this

all_RFR_thetas = [wgt_fx_list_uniq, wgt_fx_thetas, RFR_thetas1, RFR_thetas2, RFR_thetas3, RFR_thetas4]

RFR_param_df = get_alg_param_df(all_RFR_thetas, 0)

RFR_param_df.ml_theta2 = RFR_param_df.ml_theta2.astype(int) #needed to use as key in dict
RFR_param_df.ml_theta3 = RFR_param_df.ml_theta3.astype(int) #need to be int or 0-1 float


#%% Lasso
Lasso_thetas = thetas_for_penalty_str

all_lasso_thetas = [wgt_fx_list_uniq, wgt_fx_thetas, Lasso_thetas]

lasso_param_df = get_alg_param_df(all_lasso_thetas, 1)


#%% EN
EN_thetas1 = thetas_for_penalty_str # str of regularization
EN_thetas2 = np.linspace(0, 1, 15) # balance of L1/L2

all_en_thetas = [wgt_fx_list_uniq, wgt_fx_thetas, EN_thetas1, EN_thetas2]

EN_param_df = get_alg_param_df(all_en_thetas, 2)


# Combine into dict
param_df_dict = {
    'rfr': RFR_param_df,
    'lasso': lasso_param_df,
    'en': EN_param_df
    }
