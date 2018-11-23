# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:30:54 2018

@author: 14363841
"""
import numpy as np


def circular_mask(N, radius, center=None,):
    
    if center is None:
        center  =  [int(N/2), int(N/2)]
    
    Y, X = np.ogrid[:N, :N]
    dist = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    
    mask=np.zeros((N,N))
    mask[dist<=radius]=1
    return mask