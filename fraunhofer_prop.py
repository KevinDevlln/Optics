# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:21:24 2018

@author: 14363841
"""


#analytical Fraunhofer propogation
import numpy as np


def fraunhofer_prop(Uin, wvl, delta, Dz):
    
    N = Uin.shape[0]
    k = 2*np.pi/wvl
    fX = np.arange(-N/2, N/2) / (N*delta)
    
    x, y = np.meshgrid(wvl*Dz*fX, wvl*Dz*fX)
    
    return Uout, x, y