#!/usr/bin/env python

"""
The misc module contains common functions
"""

__all__ = ['skewnormal']

__version__ = '0.1.0'

__author__ = 'Sy Redding'


import numpy as np
from scipy.special import erf

def skewnormal(x, loc, scale, shape, amplitude, baseline):
    """
    skewnormal distribution
    https://en.wikipedia.org/wiki/Skew_normal_distribution
    :param x: variable
    :param loc: location parameter, mean
    :param scale: scale parameter, variance
    :param shape: shape parameter, skew
    :param amplitude:
    :param baseline:
    :return:
    """
    t = (x - loc) / scale
    pdf = 1/np.sqrt(2*np.pi) * np.exp(-t**2/2)
    cdf = (1 + erf((t*shape)/np.sqrt(2))) / 2.
    return 2*amplitude / scale * pdf * cdf + baseline