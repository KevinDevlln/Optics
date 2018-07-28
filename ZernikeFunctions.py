# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 22:18:28 2018

@author: Kevin Devlin
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Zernike_R(m, n, rho):
    
    if(np.mod(n-m, 2) ==1):
        return rho*0.0

    coeff = rho*0.0
    for k in range(int((n-m)/2+1)):
         coeff += rho**(n-2.0*k)*(-1.0)**k*math.factorial(n-k)/(math.factorial(k)*math.factorial((n+m)/2.0-k)*math.factorial((n-m)/2.0-k))
        
    return coeff


def Zernike(m, n, rho, phi, norm=True):
        
    nc = 1.0
    if (norm):
         nc = (2*(n+1)/(1+(m==0)))**0.5
         print(nc)
         
    if (m > 0): return nc*Zernike_R(m, n, rho) * np.cos(m * phi)
    if (m < 0): return nc*Zernike_R(-m, n, rho) * np.sin(-m * phi)
    else:
	return nc*Zernike_R(0, n, rho)
 

def noll_to_zern(j):
    print('Noll index: %s' % (j))
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    while (j > n):
        n += 1
        j -= n
        m = -n+2*j
    
    return (n, m)

def Zernikel(j, rho, phi, norm=True):
    n, m = noll_to_zern(j)
    print('Radial degree: %s, Azimuthal degree, %s' % (n, m))
    return Zernike(m, n, rho, phi, norm)


###############################################################################
#set up co-ordinates and grid with circular aperture
density = 256
pupilRadius = 80
x = np.arange(-density/2, density/2)
X, Y = np.meshgrid(x, x)
R = ((X**2)+(Y**2))**0.5
phi = np.arctan2(Y,X)
rho = np.ones((density, density))
R_norm = R/pupilRadius
rho[R_norm>1]=0

#add rho for circular pupil
z=Zernikel(10, R, phi)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, z ,cmap='jet')
fig.colorbar(surf, shrink=0.75)
 
# Set rotation angle to 360 degrees
ax.view_init(azim=-60)

