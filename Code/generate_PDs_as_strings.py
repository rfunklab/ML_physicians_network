# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:58:49 2022



THIS REQUIRES DATA NOT ON MSI AND PATH FIXES



@author: fpichard
"""

import os
import os.path as op
import copy
import networkx as nx
import matplotlib.pyplot as plt
import pyflagser
import numpy as np
import pandas as pd


###
#
#   Functions
#
###

def replace_inf_in_filtrations_with_max_plus_var(filtration):
    """
    Replace the infinite values in a filtration with max + 1.96*std

    Parameters
    ----------
    filtration : dict
        Results from running a filtration algorithm

    Returns
    -------
    dict
    """

    # Calc replacement value
    #one_d_array = np.array([])

    #for array in filtration["dgms"]:
        #one_d_array = np.concatenate((one_d_array, array.reshape(array.size)), axis = 0)

    #one_d_array_no_inf = np.ma.masked_array(one_d_array, ~np.isfinite(one_d_array)).filled(0)

    #max_val = one_d_array_no_inf.max()
    #std = np.std(one_d_array_no_inf)

    # New max val
    #new_max_val = max_val + 1.96*std

    # remove infs
    filtration_no_inf = copy.deepcopy(filtration)
    for i in range(0,len(filtration["dgms"])):
      filtration_no_inf["dgms"][i][filtration_no_inf["dgms"][i] == np.inf] = 1.01

    return filtration_no_inf


def generate_hsa_PD(HID, year, healthcare_type = "hsa", zero_h1 = True, zero_h2 = True):
    """
    Generate PD for a given HSA ID (HID) and year as a string
        PD str format: birth_01-death_01;birth_02-death_02...

    Parameters
    ----------
    HID : str
    year : str
    healthcare_type : str [hsa]
        hsa (default) or hrr
    zero_h1 : bool [True]
        Allow zero holes for H1
    zero_h2 : bool [True]
        Allow zero holes for H2

    Returns
    -------
    array
    """

    # Load HSA data
    hsa_dir_dict = {"TYPE": healthcare_type,
                    "SUBTYPE": "local",
                    "YEAR": year,
                    "ALG": YEAR_TO_ALG[year],
                    "ID_NUM": HID}

    hsa_file_path = REF_NETWORK_FILENAME.format_map(hsa_dir_dict)

    ## get sparse adjacency matrix
    G = nx.read_graphml(hsa_file_path)

    if np.array(G.edges).size < 5:
        return

    D = nx.to_scipy_sparse_matrix(G)


    #print("Data loaded.")

    # Get filtration
    hsa_filtration = pyflagser.flagser_weighted(adjacency_matrix=D,
                                min_dimension=0,
                                max_dimension=2,
                                directed=False,
                                coeff=2,
                                approximation=None)

    #print("Filtration complete.")

    # Fix inf
    hsa_filtration = replace_inf_in_filtrations_with_max_plus_var(hsa_filtration)

    # Get PI vecs
    h0_vals_as_str = np.array([';'.join(['-'.join(elem) for elem in hsa_filtration["dgms"][0].astype(str).tolist()])])
    
    if not hsa_filtration['dgms'][1].size == 0:
        h1_vals_as_str = np.array([';'.join(['-'.join(elem) for elem in hsa_filtration["dgms"][1].astype(str).tolist()])])

    else:
        if zero_h1:
            h1_vals_as_str = np.array([''])
        else:
            return
    
    try:
        if not hsa_filtration['dgms'][2].size == 0:
            h2_vals_as_str = np.array([';'.join(['-'.join(elem) for elem in hsa_filtration["dgms"][2].astype(str).tolist()])])

        else:
            if zero_h2:
                h2_vals_as_str = np.array([''])
            else:
                return
    except:
        if zero_h2:
            h2_vals_as_str = np.array([''])
        else:
            return

    #print("PI vectors generated.")

    # Concat all info
    hsa_all_info = np.concatenate((h0_vals_as_str, h1_vals_as_str, h2_vals_as_str))
    hsa_all_info = np.append(HID, hsa_all_info)

    return hsa_all_info


###
#
#   Config
#
###
# Paths
PROJECT_PATH = os.getcwd()
DATA_PATH    = op.join(PROJECT_PATH, "..", "..", "..", "..", "data")
REF_NET_PATH = op.join(DATA_PATH, "referral_networks", "files")
COV_PATH     = op.join(DATA_PATH, "covariates")

# Useful variables
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

REF_NETWORK_DIR = op.join(REF_NET_PATH, "{TYPE}", "{SUBTYPE}", "udw", "{YEAR}_{ALG}")

REF_NETWORK_FILENAME = op.join(REF_NETWORK_DIR, "{ID_NUM}", "{ID_NUM}_{TYPE}_{SUBTYPE}_{YEAR}_{ALG}_G.graphml.gz")



###
#
#   Main
#
###

## Prep
year = REF_NETWORK_YEARS[2]

## Extract PIs and add on outcome values
hsa_ids = np.loadtxt(op.join(PROJECT_PATH, "great_lakes_hsa_ids.txt"), dtype=str).tolist()

# Get HSA PIs
#   Creates data variable
#       Need to make sure that something is returned
data = None
for idx, hid in enumerate(hsa_ids):
    if type(data) == type(None):
        temp = generate_hsa_PD(hid, year, zero_h1 = True)
        if type(temp) != type(None):
            data = temp
    else:
        temp_data = generate_hsa_PD(hid, year, zero_h1 = True)
        if type(temp_data) != type(None):
            data = np.vstack((data, temp_data))

data = pd.DataFrame(data)

data.columns = ['hsa', 'h0', 'h1', 'h2']


## Save data
save_fn = op.join(PROJECT_PATH, "proj-PI_year-" + year + "_region-great_lakes_desc-PD_as_str.csv")

data.to_csv(save_fn, index = False)






