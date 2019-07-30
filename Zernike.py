# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 22:18:28 2018

@author: Kevin Devlin
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def zernike_R(m, n, rho):
    
    if(np.mod(n-m, 2) ==1):
        return rho*0.0

    coeff = rho*0.0
    for k in range(int((n-m)/2+1)):
         coeff += rho**(n-2.0*k)*(-1.0)**k*math.factorial(n-k)/(math.factorial(k)*math.factorial((n+m)/2.0-k)*math.factorial((n-m)/2.0-k))
        
    return coeff


def zernike(m, n, npix=100, rho=None, theta=None, norm=True, outside=np.nan, **kwargs):
        
    if theta is None and rho is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    
    aperture = np.ones(rho.shape)
    aperture[(rho > 1)] = 0.0
    
    nc = 1.0
    if (norm):
         nc = (2*(n+1)/(1+(m==0)))**0.5
         
    if (m > 0):
        z_result = nc*zernike_R(m, n, rho) * np.cos(m * theta) * aperture
    if (m < 0):
        z_result = nc*zernike_R(-m, n, rho) * np.sin(-m * theta) * aperture
    else:
        z_result = nc*zernike_R(0, n, rho)
    
    z_result[np.where(rho > 1)] = outside
    return z_result
 

def noll_to_zern(j, **kwargs):
    print('Noll index: %s' % (j))
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    while (j > n):
        n += 1
        j -= n
        m = -n+2*j
    
    return (n, m)

def Zernikel(j, **kwargs):
    n, m = noll_to_zern(j)
    print('Radial degree: %s, Azimuthal degree, %s' % (n, m))
    return zernike(m, n, **kwargs)



