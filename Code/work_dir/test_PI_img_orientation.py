#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:08:03 2022

@author: icd
"""

import sys
import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PersistenceImages.persistence_images as pimg
import PersistenceImages.weighting_fxns as wfxs

sys.path.append('./modules')

#import cv_prep_vars

def plot_pi(data, cmap = 'coolwarm', plot_type = 'imp', vval = None):
    if plot_type == 'imp':
        if vval:
            plt.pcolor(np.transpose(data), cmap=cmap, vmin = -1*vval, vmax = vval)
        else:
            plt.pcolor(np.transpose(data), cmap=cmap, vmin = -0.01, vmax = 0.01)
    else:
        if vval:
            plt.pcolor(np.transpose(data), cmap=cmap, vmin = -1*vval, vmax = vval)
        else:
            plt.pcolor(np.transpose(data), cmap=cmap)

#%% Define an imgr
mid_pixel_resolution = 0.05

pimgr = pimg.PersistenceImager(pixel_size  = mid_pixel_resolution,
                              pers_range  = (0, 1.01),
                              birth_range = (0, 1.01),
                              weight = pimg.weighting_fxns.persistence)

#%% Define test PDs

# All holes are born early and persistent very little
## These should all be at the bottom left of the image
bottom_left_pd = [[0, 0.01 + np.random.normal(loc=0.0001, scale=0.01)] for _ in range(2000)]
bottom_left_pd.append([0, 1.01]) # Append additional to make sure it shows the full range

bottom_left_pi = pimgr.transform(bottom_left_pd, skew = True)

# All holes begin early and persist till the end
## These should all be at the top left of the image
top_left_pd = [[0, 1.01] for _ in range(25)]
top_left_pd.append([0, 0.1]) # Append additional to make sure it shows the full range

top_left_pi = pimgr.transform(top_left_pd, skew = True)

# Combined
both = bottom_left_pd + top_left_pd

both_pi = pimgr.transform(np.array(both), True)

#%% Plot test cases
plot_pi(bottom_left_pi)

plot_pi(top_left_pi)

plot_pi(both_pi)
