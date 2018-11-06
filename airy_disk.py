# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:01:18 2018

@author: Kevin Devlin
"""

#function to compute the airy disk of an optical system using the anlytical
#formula


import numpy as np
from scipy.special import j1 as bessel1

def airy_disk(x, y, wave, fNumber):
    
    q = np.sqrt(x**2 + y**2)
    
    X = (np.pi*q)/wave*fNumber
    
    result = (2*bessel1(X) / X)**2
    
    try:
        # Replace value where divide-by-zero occurred with 1
        result[np.logical_or(np.isinf(result), np.isnan(result))] = 1
    except TypeError:
        # TypeError is thrown when single integers--not arrays--are passed into the function
        result = np.array([result])
        result[np.logical_or(np.isinf(result), np.isnan(result))] = 1
    
    return result
    