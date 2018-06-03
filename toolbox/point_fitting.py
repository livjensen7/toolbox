#!/usr/bin/env python

"""
The point_fitting module contains functions used in measuring single particles.
for more info see the `fitting guassians <https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/fitting-gaussians.ipynb/>`_
or the `finding local maxima <https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/finding-local-maxima.ipynb/>`_
tutorials.

"""

__all__ = ['two_d_gaussian', 'find_maxima', 'fit_routine']
__version__ = '0.0.1'
__author__ = 'Sy Redding'



import numpy as np
import skimage.filters as skim
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.optimize import curve_fit


def two_d_gaussian(span, amplitude, mu_x, mu_y, sigma_x, sigma_y, theta, offset):
    """
    Two dimensional Gaussian function, specifically for use with point-fitting functions.

    :param span: tuple containing arrays for the range of the gaussian function in x and y.
    :param amplitude: Height of the gaussian
    :param mu_x: mean in x
    :param mu_y: mean in y
    :param sigma_x: standard deviation in x
    :param sigma_y: standard deviation in y
    :param theta: Angular offset of the coordinate axis, in radians counterclockwise from x axis.
    :param offset: size of the noise floor

    :return: 1D array of values for the function across the range defined by span

    :Example:
        >>> import numpy as np
        >>> from toolbox.point_fitting import two_d_gaussian
        >>> x,y = np.linspace(0,5,6), np.linspace(0,5,6)
        >>> two_d_gaussian((x,y),1,2.5,2.5,1,1,0,.2)
        array([0.20193045, 0.30539922, 0.97880078, 0.97880078, 0.30539922,0.20193045])
    """
    (x,y) = span
    mu_x = float(mu_x)
    mu_y = float(mu_y)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-mu_x)**2) + 2*b*(x-mu_x)*(y-mu_y)
                            + c*((y-mu_y)**2)))
    return g.ravel()

def find_maxima(image,size,threshold_method = "threshold_otsu"):
    """
    Locates maxima in an image. See `here <https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/finding-local-maxima.ipynb/>`_

    :param image: 2-dimensional image array.
    :param size: int, size of the maxima and minimum filters used
    :param threshold_method: string, type of thresholding filter used. Accepts any filter in the ``skimmage.filters`` module. Default is "otsu's method"

    :return: 1D array of [(x,y),...] defining the locations of each maximum

    :Example:
        >>> import toolbox.point_fitting as pt
        >>> import toolbox.testdata as test
        >>> im = test.single_max()
        >>> print(pt.find_maxima(im,10))
        [(17, 14)]
    """
    im_max = filters.maximum_filter(image, size)
    im_min = filters.minimum_filter(image, size)
    im_diff = im_max - im_min

    maxima=(image==im_max)
    thresh = getattr(skim, threshold_method)(im_diff)
    bool_diff = (im_diff < thresh)
    maxima[bool_diff] = False

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    points = []
    for dy,dx in slices:
        points.append((dx.start,dy.start)) 
    return points

def fit_routine(Image, x, y, bbox):
    """
    Fits a 2D gaussian function to 2D image array. "x", "y", and "bbox" define an ROI fit by ``two_d_gaussian`` using
    ``scipy.optimaize.curve_fit``

    :param Image: 2D array containing ROI to be fit
    :param x: center of ROI in x
    :param x: center of ROI in y
    :param bbox: side-length of the ROI. if even, rounds up to next odd integer

    :return: 1D array of optimal parameters from gaussian fit.
    :return: ``None``

        if ROI falls (partially or otherwise) outside Image. Or, if curve_fit raises RuntimeError

    :Example:
        >>> import toolbox.point_fitting as pt
        >>> import toolbox.testdata as test
        >>> im = test.single_max()
        >>> x,y = pt.find_maxima(im,10)[0]
        >>> fit = pt.fit_routine(im, x, y, 10)
        >>> print(Fit)
        array([ 1.01462278, 16.74464408, 14.28216362,  0.78958828,  1.14957256,
        2.25853845,  0.11157521])
    """
    db = int(np.floor(bbox/2))
    bbox = bbox+(1-bbox%2)
    span_x = np.linspace(x-db,x+db, bbox)
    span_y = np.linspace(y-db,y+db, bbox)
    X,Y = np.meshgrid(span_x, span_y)
    if 0<= y-db <= y+db+1 <= Image.shape[0] and 0<= x-db <= x+db+1 <= Image.shape[1]:
        pixel_vals = Image[y-db:y+db+1, x-db:x+db+1].ravel()
        scaled = [k/max(pixel_vals) for k in pixel_vals]
        initial_guess = (1, x, y, 1, 1, 0, 0)
        try:
            popt, pcov = curve_fit(two_d_gaussian, (X, Y), scaled, p0=initial_guess)
        except RuntimeError:
            popt = None
    else:
        popt = None
    return popt