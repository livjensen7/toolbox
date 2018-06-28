#!/usr/bin/env python

"""
The alignment module contains functions used in aligning two channel
data. See our `walkthrough <https://github.com/ReddingLab/Learning/blob
/master/image-analysis-basics/Image-alignment-with-toolbox.ipynb/>`_
of the alignment module's usage.
"""

__all__ = ['im_split', 'get_offset_distribution',
           'plot_assigned_maxima', 'inspect_global_fit',
           'inspect_individual_fits', 'align_by_offset',
           'overlay']

__version__ = '0.1.0'

__author__ = 'Sy Redding and Liv Jensen'





import numpy as np
import random as ra
import matplotlib.pyplot as plt
from toolbox.misc import skewnormal
from toolbox.point_fitting import find_maxima, fit_routine
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from skimage.transform import warp_coords, rotate


########################################################################

def clean_duplicate_maxima(dist, indexes):
    paired_indexes = []
    count = 0
    for i in set(indexes):
        tmp = [np.inf, np.inf]
        for j, k in zip(indexes, dist):
            if i == j and k < abs(tmp[1]):
                tmp = [j, count]
                count += 1
            elif i == j:
                count += 1
            else:
                pass
        paired_indexes.append(tmp)
    return paired_indexes


def make_bins(data, width):
    return np.arange(min(data), max(data) + width, width)

def find_global_offset(im_stack, bbox=9, splitstyle="hsplit",
                       fsize=10, binwidth=.1):

    pooled_x, pooled_y = [], []
    for im in im_stack:
        xdist, ydist = get_offset_distribution(im, bbox, splitstyle,
                                               fsize)
        pooled_x += xdist
        pooled_y += ydist

    p0 = [bincens[np.argmax(vals[0])],.2, 1, max(vals[0]), 0]
    bins = make_bins(pooled_x, binwidth)
    vals = np.histogram(pooled_x, bins)
    bincens = [bins[j] + (binwidth / 2.) for j in range(len(bins) - 1)]
    popt_x, pcov_x = curve_fit(skewnormal, np.array(bincens),
                               np.array(vals[0]),p0)

    bins = make_bins(pooled_y, binwidth)
    vals = np.histogram(pooled_y, bins)
    bincens = [bins[j] + (binwidth / 2.) for j in range(len(bins) - 1)]
    p0 = [bincens[np.argmax(vals[0])],.2, 1, max(vals[0]), 0]
    popt_y, pcov_y = curve_fit(skewnormal, np.array(bincens),
                               np.array(vals[0]),p0)

    return popt_x[0], popt_y[0]


########################################################################


def im_split(Image, splitstyle="hsplit"):
    """
    Image passed to this function is split into two channels based on
    "splitstyle".
    ***note*** micromanager images and numpy arrays are indexed
    opposite of one another.

    :param Image: 2D image array
    :param splitstyle: str, accepts "hsplit", "vsplit". Default is
    "hsplit"

    :return: The two subarrays of Image split along specified axis.

    :Example:

        >>> from toolbox.alignment import im_split
        >>> import toolbox.testdata as test
        >>> im = test.image_stack()[0]
        >>> ch1, ch2 = im_split(im)
        >>> ch1.shape, ch2.shape
        ((512, 256), (512, 256))
        >>> ch1, ch2 = im_split(im, "vsplit")
        >>> ch1.shape, ch2.shape
        ((256, 512), (256, 512))
    """
    return getattr(np, splitstyle)(Image, 2)[0], \
           getattr(np, splitstyle)(Image, 2)[1]





def get_offset_distribution(Image, bbox=9, splitstyle="hsplit",
                            fsize=10):
    """
    This function in order:
        * splits the image into channels
        * locates and fits all of the points in each channel
        * pairs up associated points from each channel, uses cDKTree
        * and determines their offset

    :param Image: 2D image array
    :param bbox: int, passed to ``point_fitting.fit_routine``,
    size of ROI around each point to apply gaussian fit. Default is 9.
    :param splitstyle: string, passed to ``im_split``; accepts
    "hsplit", "vsplit". Default is "hsplit"
    :param fsize: int, passed to ``point_fitting.find_maxima``,
    size of average filters used in maxima determination. Default is 10.

    :return: Two lists containing all of the measured x- and y- offsets

    :Example:

        >>> from toolbox.alignment import get_offset_distribution
        >>> import toolbox.testdata as test
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> im = test.image_stack()[0]
        >>> x_dist, y_dist = get_offset_distribution(im)
        >>> print(np.mean(x_dist), np.mean(y_dist))
        -1.9008888233326608 -2.042675546813981
    """
    ch1, ch2 = im_split(Image, splitstyle)
    ch1_maxima = find_maxima(ch1, fsize)
    ch2_maxima = find_maxima(ch2, fsize)
    Delta_x, Delta_y = [], []
    mytree = cKDTree(ch1_maxima)
    dist, indexes = mytree.query(ch2_maxima)
    for i, j in clean_duplicate_maxima(dist, indexes):
        x1, y1 = ch1_maxima[i]
        x2, y2 = ch2_maxima[j]
        fit_ch1 = fit_routine(ch1, x1, y1, bbox)
        fit_ch2 = fit_routine(ch2, x2, y2, bbox)
        try:
            Delta_x.append(fit_ch1[1] - fit_ch2[1])
            Delta_y.append(fit_ch1[2] - fit_ch2[2])

        except TypeError:
            pass
    return (Delta_x, Delta_y)





def plot_assigned_maxima(Image, splitstyle="hsplit", fsize=10):
    """
    This function spits out a matplotlib plot with lines drawn
    between each of the assigned pairs of maxima.
    The purpose of this function is more for a sanity check than
    anything useful.

    :param Image: 2D image array
    :param splitstyle: string, passed to ``im_split``; accepts
    "hsplit", "vsplit". Default is "hsplit"
    :param fsize: int, passed to ``point_fitting.find_maxima``,
    size of average filters used in maxima determination. Default is 10.

    :return: fancy plot of assigned points.

    :Example:

        >>> from toolbox.alignment import plot_assigned_maxima
        >>> import toolbox.testdata as test
        >>> im = test.image_stack()[0]
        >>> plot_assigned_maxima(im)
    """
    ch1, ch2 = im_split(Image, splitstyle)
    ch1_maxima = find_maxima(ch1, fsize)
    ch2_maxima = find_maxima(ch2, fsize)
    width = ch2.shape[1]
    plt.figure(figsize=(Image.shape[0] / 64, Image.shape[1] / 64))
    plt.axis('off')
    plt.imshow(Image, cmap="binary_r")
    plt.title("Assigned matching points")

    mytree = cKDTree(ch1_maxima)
    dist, indexes = mytree.query(ch2_maxima)
    for i, j in clean_duplicate_maxima(dist, indexes):
        x1, y1 = ch1_maxima[i]
        x2, y2 = ch2_maxima[j]
        tmp_color = (
        ra.uniform(0, 1), ra.uniform(0, 1), ra.uniform(0, 1))
        plt.plot(x1, y1, color=tmp_color, marker='+')
        plt.plot(x2 + width, y2, color=tmp_color, marker='+')
        plt.plot([x1, x2 + width], [y1, y2], color=tmp_color)
    plt.show()





def inspect_global_fit(im_stack, bbox=9, fsize=10,
                       binwidth=.1, init_params = None,
                       splitstyle="hsplit",showplot=True):

    """
    Basic alignment function. Accepts a 1D list of image arrays,
    then splits the images, locates corresponding maxima in each
    channel and then calculates the best shift in x and y to align
    each maxima pair. If showplot is set to True, this function also
    produces a pair of histograms of all the measured offsets and
    resulting fits to those data.


    :param im_stack: 1D list of image arrays to be used in
    determination of the offset
    :param bbox: int, passed to ``point_fitting.fit_routine``,
    size of ROI around each point to apply gaussian fit. Default is 9.
    :param fsize: int, passed to ``point_fitting.find_maxima``,
    size of average filters used in maxima determination. Default is 10.
    :param binwidth: float, passed to ``make_bins``; resolution of
    histogram for fitting
    :param init_params: 1D array, initial conditions passed to
    scipy.optimize.curve_fit. must be length 5,
    p0 = [loc, scale, shape, amplitude, baseline]
    :param splitstyle: string, passed to ``im_split``; orientation
    of channels, vertical or horizontal
    :param showplot:  bool, if True, will generate a plot of the
    distribution and fit

    :return: tuple containing optimal parameters and covariance
    matrix from fit. (popt_x, pcov_x, popt_y, pcov_y)

    :Example:

        >>> from toolbox.alignment import inspect_global_fit
        >>> import toolbox.testdata as test
        >>> params = inspect_global_fit(test.image_stack())
        >>> print(params[0][0],params[2][0])
        5.612082237088681 -2.651765063702885
    """
    ###
    pooled_x, pooled_y = [], []
    for im in im_stack:
        xdist, ydist = get_offset_distribution(im, bbox, splitstyle,
                                               fsize)
        pooled_x += xdist
        pooled_y += ydist

    ###
    bins = make_bins(pooled_x, binwidth)
    x_bincens = [bins[j] + (binwidth / 2.) for j in
                 range(len(bins) - 1)]
    x_vals = np.histogram(pooled_x, bins)

    if init_params is None:
        p0 = [x_bincens[np.argmax(x_vals[0])],
              .2, 1, max(x_vals[0]), 0]

    try:
        popt_x, pcov_x = curve_fit(skewnormal, np.array(x_bincens),
                                   np.array(x_vals[0]),p0)
    except RuntimeError:
        popt_x, pcov_x = [], []
        pass

    ###
    bins = make_bins(pooled_y, binwidth)
    y_bincens = [bins[j] + (binwidth / 2.) for j in
                 range(len(bins) - 1)]
    y_vals = np.histogram(pooled_y, bins)

    if init_params is None:
        p0 = [y_bincens[np.argmax(y_vals[0])],
              .2, 1, max(y_vals[0]), 0]

    try:
        popt_y, pcov_y = curve_fit(skewnormal, np.array(y_bincens),
                                   np.array(y_vals[0]),p0)
    except RuntimeError:
        popt_y, pcov_y = [], []
        pass

    ###
    if showplot == True:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_xlim(x_bincens[np.argmax(x_vals[0])] - 1.5,
                     x_bincens[np.argmax(x_vals[0])] + 1.5)
        ax1.set_title("x-offsets")
        ax1.bar(x_bincens, x_vals[0], width=binwidth / 2,
                color="#008fd5")
        if len(popt_x) > 1:
            fit = skewnormal(np.array(x_bincens), *popt_x)
            ax1.plot(x_bincens, fit, "--", color="#fc4f30",
                     linewidth=3)

        ax2.set_xlim(y_bincens[np.argmax(y_vals[0])] - 1.5,
                     y_bincens[np.argmax(y_vals[0])] + 1.5)
        ax2.set_title("y-offsets")
        ax2.bar(y_bincens, y_vals[0], width=binwidth / 2,
                color="#6d904f")
        if len(popt_y) > 1:
            fit = skewnormal(np.array(y_bincens), *popt_y)
            ax2.plot(y_bincens, fit, "--", color="#e5ae38",
                     linewidth=3)

        plt.show()

    return ((popt_x[0],popt_y[0],[popt_x, pcov_x, popt_y, pcov_y]))


def inspect_individual_fits(im_stack, bbox=9, fsize=10,
                            binwidth=.1, init_params = None,
                            splitstyle="hsplit"):

    """
    This function provides a method to plot each individual offset
    distributions of images passed to the function. Common usage is
    to get a sense if all the images are together.

    :param im_stack: 1D list of image arrays to be used in
    determination of the offset
    :param bbox: int, passed to ``point_fitting.fit_routine``,
    size of ROI around each point to apply gaussian fit. Default is 9.
    :param fsize: int, passed to ``point_fitting.find_maxima``,
    size of average filters used in maxima determination. Default is 10.
    :param binwidth: float, passed to ``make_bins``; resolution of
    histogram for fitting
    :param init_params: 1D array, initial conditions passed to
    scipy.optimize.curve_fit. must be length 5,
    p0 = [loc, scale, shape, amplitude, baseline]
    :param splitstyle: string, passed to ``im_split``; orientation
    of channels, vertical or horizontal


    :return: produces a plot of histograms and fits for each
    individual image passed in im_stack. Also returns list of tuples,
    each contains optimal parameters and covariance matrix from fit.
    list alternates x and y-offset fits.

    :Example:
        >>> from toolbox.alignment import inspect_individual_fits
        >>> import toolbox.testdata as test
        >>> params = inspect_individual_fits(test.image_stack())
    """
    pooled_x, pooled_y = [], []
    for im in im_stack:
        xdist, ydist = get_offset_distribution(im)
        pooled_x += xdist
        pooled_y += ydist
    spanx = [np.median(pooled_x) - 1.5, np.median(pooled_x) + 1.5]
    spany = [np.median(pooled_y) - 1.5, np.median(pooled_y) + 1.5]

    fig, axes = plt.subplots(nrows=len(im_stack), ncols=2)
    plt.subplots_adjust(wspace=0, hspace=0)
    axlist = [i for i in axes.flat]

    count = 0
    fitlist = []
    for im in im_stack:
        xdist, ydist = get_offset_distribution(im, bbox, splitstyle,
                                               fsize)

        bins = make_bins(xdist, binwidth)
        x_bincens = [bins[j] + (binwidth / 2.) for j in
                     range(len(bins) - 1)]
        x_vals = np.histogram(xdist, bins)
        axlist[count].set_xlim(spanx)
        axlist[count].axes.get_yaxis().set_visible(False)
        axlist[count].bar(x_bincens, x_vals[0], width=binwidth / 2,
                          color="#008fd5")

        if init_params is None:
            p0 = [x_bincens[np.argmax(x_vals[0])],
                  .2, 1, max(x_vals[0]), 0]
        try:
            popt_x, pcov_x = curve_fit(skewnormal, np.array(x_bincens),
                                       np.array(x_vals[0]),p0)
            fitlist.append((popt_x, pcov_x))
            fit = skewnormal(np.array(x_bincens), *popt_x)
            axlist[count].plot(x_bincens, fit, "--",
                               color="#fc4f30", linewidth=2)

        except RuntimeError:
            fitlist.append([])
            pass

        count += 1
        bins = make_bins(ydist, binwidth)
        y_bincens = [bins[j] + (binwidth / 2.) for j in
                     range(len(bins) - 1)]
        y_vals = np.histogram(ydist, bins)
        axlist[count].set_xlim(spany)
        axlist[count].axes.get_yaxis().set_visible(False)
        axlist[count].bar(y_bincens, y_vals[0], width=binwidth / 2,
                          color="#6d904f", linewidth=2)

        if init_params is None:
            p0 = [y_bincens[np.argmax(y_vals[0])],
                  .2, 1, max(y_vals[0]), 0]
        try:
            popt_y, pcov_y = curve_fit(skewnormal, np.array(y_bincens),
                                       np.array(y_vals[0]), p0)
            fitlist.append((popt_y, pcov_y))
            fit = skewnormal(np.array(y_bincens), *popt_y)
            axlist[count].plot(y_bincens, fit, "--",
                               color="#e5ae38")

        except RuntimeError:
            fitlist.append([])
            pass

        count += 1
    plt.show()

    return (fitlist)



def align_by_offset(Image, shift_x, shift_y, splitstyle="hsplit",
                    shift_channel=1):
    """
    This function shifts one channel of the array based supplied
    offset values. Retains the single image structure.

    :param Image: 2D image array
    :param shift_x: float, offset in x
    :param shift_y: float, offset in y
    :param splitstyle: string, passed to ``im_split``; accepts
    "hsplit", "vsplit". Default is "hsplit"
    :param shift_channel: int, which channel to shift by offsets,
    default is channel 1.

    :return: 2D image array of aligned image

    :Example:
        >>> from toolbox.alignment import find_global_offset,
        align_by_offset
        >>> import toolbox.testdata as test
        >>> import matplotlib.pyplot as plt
        >>> im = test.image_stack()
        >>> dx, dy = find_global_offset(im)
        >>> new_image = align_by_offset(im[0], dx, dy)
        >>> plt.imshow(new_image), plt.show()
    """

    ch1, ch2 = im_split(Image, splitstyle)
    if shift_channel == 1:
        new_coords = warp_coords(
            lambda xy: xy - np.array([shift_x, shift_y]), ch2.shape)
        warped_channel = map_coordinates(ch2, new_coords)
        aligned_image = np.concatenate((ch1, warped_channel), axis=1)
    else:
        new_coords = warp_coords(
            lambda xy: xy + np.array([shift_x, shift_y]), ch1.shape)
        warped_channel = map_coordinates(ch1, new_coords)
        aligned_image = np.concatenate((warped_channel, ch2), axis=1)
    return aligned_image


def overlay(Image, splitstyle="hsplit", rot=True, invert=False):
    """
    Overlays the two channels derived from Image. Converts Image to
    an 8-bit RGB array, with one channel colored magenta and the
    other green.

    :param Image: 2D image array
    :param splitstyle: string, passed to ``im_split``; accepts
    "hsplit", "vsplit". Default is "hsplit"
    :param rot: bool, if True, image is rotated 90 degrees
    :param invert: bool, if True, inverts the channel color assignment.
    :return: 8-bit RGB image

    :Example:
        >>> from toolbox.alignment import overlay
        >>> import toolbox.testdata as test
        >>> import matplotlib.pyplot as plt
        >>> im = test.image_stack()
        >>> dx, dy = find_global_offset(im)
        >>> aligned_image = align_by_offset(im[0], dx, dy)
        >>> overlayed = overlay(aligned_image)
        >>> plt.imshow(overlayed), plt.show()
    """
    if not invert:
        ch1, ch2 = im_split(Image,splitstyle)
    else:
        ch2, ch1 = im_split(Image,splitstyle)
    ch1_max = ch1.max()
    ch2_max = ch2.max()
    shape = ch1.shape
    red = np.zeros(shape)
    green = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            red[x, y] = ch1[x, y] / ch1_max
            green[x, y] = ch2[x, y] / ch2_max
    rgb_stack = np.dstack((red, green, red))
    if rot:
        rgb_stack = rotate(rgb_stack, -90, resize=True)

    rgb_stack *= 255
    rgb_stack = rgb_stack.astype(np.uint8)
    return rgb_stack

