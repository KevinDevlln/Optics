# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:37:46 2018

@author: 14363841
"""
import numpy as np


'''
properly scaled fourier transform functions
    
    g (array): input array for fft
    delta (float): element spacing
    
    returns:
        scaled array fft
                
'''


def ft(g, delta):
    
    G = np.fft.fftshift(np.fft.fft(
            np.fft.fftshift(g, axes=(-1))),
                    axes=(-1)) * delta
    
    return G
    

def ift(G, delta_f):
    
    g = np.fft.ifftshift(
            np.fft.ifft(np.fft.ifftshift(
                    G, axes=(-1))), axes=(-1)) *len(G) * delta_f
    
    return g


def ft2(g, delta):
    
    G = np.fft.fftshift(
            np.fft.fft2(
                    np.fft.fftshift(g, axes=(-1, -2))
                    ), axes=(-1, -2)) * delta**2
    
    return G


def if2(G, delta_f):
    
    N = G.shape[0]
    g = np.fft.ifftshift(
            np.fft.ifft2(
                    np.fft.ifftshift(G))) * (N * delta_f)**2
    
    return g