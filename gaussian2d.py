# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:19:44 2018

@author: 14363841
"""
import numpy
import matplotlib.pyplot as plt


def gaussian2d(size, radius, amplitude=1., cent=None):
    '''
    Generates 2D gaussian distribution in an NxN array
    Input:
    size : tuple, float
        Dimensions of Array to place gaussian (y, x)
    radius : float
        radius of Gaussian mask
    amplitude : float
        Amplitude of guassian distribution
    cent : tuple
        Centre of distribution on grid in order (y, x).
        
    Output:
    image : array
        2d array containing gaussian mask
    '''
        
    if not cent:
        xCent = Size/2.
        yCent = Size/2.
    else:
        yCent = cent[0]
        xCent = cent[1]

    X, Y = numpy.meshgrid(range(0, Size), range(0, Size))

    image = amplitude * numpy.exp(
        -(((xCent - X) ** 2) + ((yCent - Y)) ** 2) / (2 * radius))

    return image


if __name__ == "__main__":
    
    pass

    g = gaussian2d(21, 5, cent=(10,10))
    
    plt.figure()
    plt.imshow(g)
    plt.colorbar()
