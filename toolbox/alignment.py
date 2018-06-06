#!/usr/bin/env python

"""
The alignment module contains functions used in aligning two channel data.
See our `walkthrough <https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/Image-alignment-with-toolbox.ipynb/>`_
of the alignment module's usage.
"""

__all__ = ['FD_rule_bins', 'scrub_outliers', 'im_split',
           'get_offset_distribution', 'find_global_offset',
           'plot_assigned_maxima','align_by_offset','overlay']
__version__ = '0.0.1'
__author__ = 'Sy Redding'

import numpy as np
import random as ra
import matplotlib.pyplot as plt
from toolbox.point_fitting import find_maxima, fit_routine
from scipy.stats import skewnorm
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from skimage.transform import warp_coords,rotate


def FD_rule_bins(data):
    """
    Finds the optimal spacing of histogram bins based on the
    Freedman-Diaconis rule. https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule

    :param data: 1D array of data points

    :return: 1D array of bin edges. passes directly to numpy.histogram or matplotlib.pyplot.hist

    :Example:

        >>> import numpy as np
        >>> from toolbox.alignment import FD_rule_bins
        >>> x = np.random.normal(size=100)
        >>> FD_rule_bins(x)
        array([-2.25503346, -1.75181888, -1.2486043 , -0.74538972, -0.24217514,
                0.26103944,  0.76425402,  1.2674686 ,  1.77068318,  2.27389776,
                2.77711234])
    """
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    opt_binwidth = 2*iqr/(len(data)**(1/3.))
    return np.arange(min(data), max(data) + opt_binwidth, opt_binwidth)


def scrub_outliers(data):
    """
    Removes outliers from data. Works in two steps:
        * First, data is binned using ``FD_rule_bins`` and only the most highly populated bins are retained
        * Second, any datum more than two standard deviations away from the mean are filtered out

    :param data: 1D array or list of data points

    :return: Filtered result, 1D array

    :Example:

        >>> import numpy as np
        >>> from toolbox.alignment import scrub_outliers
        >>> x = np.concatenate((np.random.normal(size=200),np.random.uniform(-10,10,size=20)))
        >>> scrubed_x = scrub_outliers(x)
        >>> len(x), len(scrubed_x)
        (220, 179)
        >>> import matplotlib.pyplot as plt
        >>> from toolbox.alignment import FD_rule_bins
        >>> plt.figure()
        >>> plt.hist(x, FD_rule_bins(x), fc = "m")
        >>> plt.hist(scrubed_x, FD_rule_bins(x), fc = "g")
        >>> plt.show()
    """
    vals = np.histogram(data, FD_rule_bins(data))
    sorted_counts = sorted(vals[0])
    binslist = [i for i in sorted_counts if i > .9 * sorted_counts[-1]]

    # -initial scrub using taking just highly populated bins
    scrubbed_data = []
    for i in binslist:
        leftedge = vals[0].tolist().index(i)
        for datum in data:
            if datum < vals[1][leftedge + 1] and datum > vals[1][leftedge]:
                scrubbed_data.append(datum)

    # -final scrub using standard deviation
    scrubbed_data = [datum for datum in scrubbed_data if
                     datum < np.mean(scrubbed_data) + 2 * np.std(scrubbed_data) and
                     datum > np.mean(scrubbed_data) - 2 * np.std(scrubbed_data)]
    return scrubbed_data


def clean_duplicate_maxima(dist, indexes):
    paired_indexes = []
    count = 0
    for i in set(indexes):
        tmp = [np.inf,np.inf]
        for j,k in zip(indexes, dist):
            if i == j and k < abs(tmp[1]):
                tmp = [j,count]
                count += 1
            elif i == j:
                count += 1
            else:
                pass
        paired_indexes.append(tmp)
    return paired_indexes


def im_split(Image, splitstyle = "hsplit"):
    """
    Image passed to this function is split into two channels based on "splitstyle".
    ***note*** micromanager images and numpy arrays are indexed opposite of one another.

    :param Image: 2D image array
    :param splitstyle: str, accepts "hsplit", "vsplit". Default is "hsplit"

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
    return getattr(np, splitstyle)(Image, 2)[0],getattr(np, splitstyle)(Image, 2)[1]


def get_offset_distribution(Image, bbox = 9, splitstyle="hsplit", fsize=10):
    """
    This function in order:
        * splits the image into channels
        * locates and fits all of the points in each channel
        * pairs up associated points from each channel, uses cDKTree
        * and determines their offset

    :param Image: 2D image array
    :param bbox: int, passed to ``point_fitting.fit_routine``, size of ROI around each point to apply gaussian fit. Default is 9.
    :param splitstyle: string, passed to ``im_split``; accepts "hsplit", "vsplit". Default is "hsplit"
    :param fsize: int, passed to ``point_fitting.find_maxima``, size of average filters used in maxima determination. Default is 10.

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
            Delta_x.append(fit_ch1[1]-fit_ch2[1])
            Delta_y.append(fit_ch1[2]-fit_ch2[2])
            
        except TypeError:
            pass
    return Delta_x, Delta_y, "thisislocal"


def find_global_offset(im_list, bbox=9, splitstyle="hsplit", fsize=10):
    """
    This function finds the optimal x-offset and y-offset of the data using ``scrub_outliers`` to filter
    the data collected from ``get_offset_distribution``. The filtered data are then fit using ``scipy.stats.skewnorm``

    :param im_list: 1D list of image arrays to be used in determination of the offset
    :param bbox: int, passed to ``point_fitting.fit_routine``, size of ROI around each point to apply gaussian fit. Default is 9.
    :param splitstyle: string, passed to ``im_split``; accepts "hsplit", "vsplit". Default is "hsplit"
    :param fsize: int, passed to ``point_fitting.find_maxima``, size of average filters used in maxima determination. Default is 10.

    :return: Mean x- and y-offset values.

    :Example:
        >>> from toolbox.alignment import find_global_offset
        >>> import toolbox.testdata as test
        >>> im = test.image_stack()
        >>> print(find_global_offset(im))
        (5.624042070667237, -2.651128775580636)
    """
    pooled_x, pooled_y = [], []
    for im in im_list:
        xdist, ydist = get_offset_distribution(im, bbox, splitstyle, fsize)
        pooled_x += scrub_outliers(xdist)
        pooled_y += scrub_outliers(ydist)
    skew, mu1, sigma1 = skewnorm.fit(pooled_x)
    skew, mu2, sigma2 = skewnorm.fit(pooled_y)
    return mu1, mu2


def plot_assigned_maxima(Image, splitstyle="hsplit", fsize=10):
    """
    This function spits out a matplotlib plot with lines drawn between each of the assigned pairs of maxima.
    The purpose of this function is more for a sanity check than anything useful.

    :param Image: 2D image array
    :param splitstyle: string, passed to ``im_split``; accepts "hsplit", "vsplit". Default is "hsplit"
    :param fsize: int, passed to ``point_fitting.find_maxima``, size of average filters used in maxima determination. Default is 10.

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
    fig = plt.figure(figsize=(Image.shape[0]/64,Image.shape[1]/64))
    plt.axis('off')
    plt.imshow(Image, cmap="binary_r")
    plt.title("Assigned matching points")

    mytree = cKDTree(ch1_maxima)
    dist, indexes = mytree.query(ch2_maxima)
    for i, j in clean_duplicate_maxima(dist, indexes):
        x1, y1 = ch1_maxima[i]
        x2, y2 = ch2_maxima[j]
        tmp_color = (ra.uniform(0, 1), ra.uniform(0, 1), ra.uniform(0, 1))
        plt.plot(x1, y1, color=tmp_color, marker='+')
        plt.plot(x2+width, y2, color=tmp_color, marker='+')
        plt.plot([x1, x2+width], [y1, y2], color=tmp_color)
    plt.show()


def align_by_offset(Image, shift_x, shift_y, splitstyle="hsplit", shift_channel = 1):
    """
    This function shifts one channel of the array based supplied offset values. Retains the single image
    structure.

    :param Image: 2D image array
    :param shift_x: float, offset in x
    :param shift_y: float, offset in y
    :param splitstyle: string, passed to ``im_split``; accepts "hsplit", "vsplit". Default is "hsplit"
    :param shift_channel: int, which channel to shift by offsets, default is channel 1.

    :return: 2D image array of aligned image

    :Example:
        >>> from toolbox.alignment import find_global_offset, align_by_offset
        >>> import toolbox.testdata as test
        >>> import matplotlib.pyplot as plt
        >>> im = test.image_stack()
        >>> dx, dy = find_global_offset(im)
        >>> new_image = align_by_offset(im[0], dx, dy)
        >>> plt.imshow(new_image), plt.show()
    """
    if splitstyle == "vsplit":
        ch2, ch1 = im_split(Image, splitstyle)
    else:
        ch1, ch2 = im_split(Image, splitstyle)
    if shift_channel == 1:
        new_coords = warp_coords(lambda xy: xy - np.array([shift_x, shift_y]), ch2.shape)
        warped_channel = map_coordinates(ch2, new_coords)
        aligned_image = np.concatenate((ch1, warped_channel), axis=1)
    else:
        new_coords = warp_coords(lambda xy: xy + np.array([shift_x, shift_y]), ch1.shape)
        warped_channel = map_coordinates(ch1, new_coords)
        aligned_image = np.concatenate((warped_channel, ch2), axis=1)
    return aligned_image


def overlay(Image, rot=True, invert=False):
    """
    Overlays the two channels derived from Image. Converts Image to an 8-bit RGB array, with one channel colored magenta and the other green.

    :param Image: 2D image array
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
        ch1, ch2 = im_split(Image)
    else:
        ch2, ch1 = im_split(Image)
    ch1_max = ch1.max()
    ch2_max = ch2.max()
    shape = ch1.shape
    red = np.zeros(shape)
    green = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            red[x, y] = ch1[x, y]/ch1_max
            green[x, y] = ch2[x, y]/ch2_max
    rgb_stack = np.dstack((red, green, red))
    if rot:
        rgb_stack = rotate(rgb_stack, -90, resize=True)

    rgb_stack *= 255
    rgb_stack = rgb_stack.astype(np.uint8)
    return rgb_stack
