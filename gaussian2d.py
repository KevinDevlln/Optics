# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:19:44 2018

@author: 14363841
"""
import numpy
import matplotlib.pyplot as plt


def gaussian2d(size, radius, amplitude=1., cent=None):
    '''
    Generates 2D gaussian distribution
    Args:
        size (tuple, float): Dimensions of Array to place gaussian (y, x)
        radius: radius of Gaussian mask
        amplitude (float): Amplitude of guassian distribution
        cent (tuple): Centre of distribution on grid in order (y, x).
    '''

    try:
        xSize = size[0]
        ySize = size[1]
    except (TypeError, IndexError):
        xSize = ySize = size
        
    if not cent:
        xCent = xSize/2.
        yCent = ySize/2.
    else:
        yCent = cent[0]
        xCent = cent[1]

    X, Y = numpy.meshgrid(range(0, xSize), range(0, ySize))

    image = amplitude * numpy.exp(
        -(((xCent - X) ** 2) + ((yCent - Y)) ** 2) / (2 * radius))

    return image


if __name__ == "__main__":
    
    pass

    g = gaussian2d(21, 5, cent=(10,10))
    
    plt.figure()
    plt.imshow(g)
    plt.colorbar()