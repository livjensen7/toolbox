import numpy as np
import random as ra
import matplotlib.pyplot as plt
from toolbox.point_fitting import findMaxima, fitRoutine
from scipy.stats import norm
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from skimage.transform import warp_coords




def FD_rule_bins(data):
    """
    Finds the optimal spacing of histogram bins based on the
    Freedman-Diaconis rule. https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule

    :param data: 1D array of data points

    :return: 1D array of bin edges. passes directly to numpy.histogram or matplotlib.plot.hist

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


def scrub_outliers(data,recurse = 2):
    """
    Removes outliers from data based on standard deviation.
    if data point is more than two standard deviations away from the mean
    it is removed. Process is iterative.

    :param data: 1D array or list of data points
    :param recurse: Number of times to iterate over the process. Default is twice.

    :return: New 1D list of data without outliers.

    :Example:

        >>> import numpy as np
        >>> from toolbox.alignment import scrub_outliers
        >>> x = np.concatenate((np.random.normal(size=200),np.random.uniform(-10,10,size=20)))
        >>> scrubed_x = scrub_outliers(x)
        >>> len(x),len(scrubed_x)
        (220, 196)
        >>> import matplotlib.pyplot as plt
        >>> from toolbox.alignment import FD_rule_bins
        >>> plt.figure()
        >>> plt.hist(x,FD_rule_bins(x),fc = "m")
        >>> plt.hist(scrubed_x,FD_rule_bins(x), fc = "g")
        >>> plt.show()
    """

    recur_count = 0
    while recur_count<recurse:
        recur_count += 1
        data = [datum for datum in data if
                datum < np.mean(data) + 2*np.std(data) and
                datum > np.mean(data) - 2*np.std(data)]
    return data



def get_offset_distribution(Image,bbox = 9):
    """
    Image passed to this function should be 2-channel data divided vertically.
    This function in order:
        * splits the image into left and right channels
        * locates and fits all of the foci in each channel
        * pairs up associated foci from each channel and determines their x- and y- offset

    :param Image: 2D image array
    :param bbox: int, size of ROI around each point to apply gaussian fit.

    :return: Two lists containing the x- and y- offsets of each corresponding pair of foci.

    :Example:

        >>> import toolbox.alignment as al
        >>> import toolbox.testdata as test
        >>> import matplotlib.pyplot as plt
        >>> im = test.image_stack()[0]
        >>> x_dist,y_dist = al.get_offset_distribution(im)
        >>> plt.hist(x_dist),plt.hist(y_dist)
        >>> plt.show()
    """
    leftch_maxima = findMaxima(np.hsplit(Image, 2)[0],10)
    rightch_maxima = findMaxima(np.hsplit(Image, 2)[1],10)
    Delta_x,Delta_y = [],[]
    mytree = cKDTree(leftch_maxima)
    dist, indexes = mytree.query(rightch_maxima)
    for i in range(len(leftch_maxima)):
        print(len(leftch_maxima), len(indexes))
        x1, y1 = leftch_maxima[indexes[i]]
        x2, y2 = rightch_maxima[i]
        fit_ch1 = fitRoutine(np.hsplit(Image, 2)[0], x1, y1, bbox)
        fit_ch2 = fitRoutine(np.hsplit(Image, 2)[1], x2, y2, bbox)
        try:
            Delta_x.append(fit_ch1[1]-fit_ch2[1])
            Delta_y.append(fit_ch1[2]-fit_ch2[2])
            
        except TypeError:
            pass
    return(Delta_x,Delta_y)

def find_global_offset(im_list, bbox = 9):
    """
    finds the optimal x-shift and y-shift of the data.

    :param im_list: 1D list of image arrays used in determination of the offset
    :param bbox: int, size of ROI around each point to apply gaussian fit.

    :return: Mean x and y shift values to align all images best fit.

    :Example:

        >>> import toolbox.alignment as al
        >>> import toolbox.testdata as test
        >>> im = test.image_stack()
        >>> print(al.find_global_offset(im, 8,3,7))
        (5.860408756886414, -2.7643538511181185)
    """
    pooled_x, pooled_y = [], []
    for im in im_list:
        xdist, ydist = get_offset_distribution(im, bbox)
        pooled_x += scrub_outliers(xdist)
        pooled_y += scrub_outliers(ydist)
    mu1, sigma1 = norm.fit(pooled_x)
    mu2, sigma2 = norm.fit(pooled_y)
    return mu1, mu2


def plot_assigned_maxima(Image):
    """
    plots the assigned maxima from each channel. Uses cKDTree

    :param Image: 2D image array

    :return: plot of assigned points.

    :Example:

        >>> import toolbox.alignment as al
        >>> import toolbox.testdata as test
        >>> im = test.image_stack()[0]
        >>> al.plot_assigned_maxima(im,8,3)
    """
    leftch_maxima = findMaxima(np.hsplit(Image, 2)[0],10)
    rightch_maxima = findMaxima(np.hsplit(Image, 2)[1],10)
    width = np.hsplit(Image, 2)[1].shape[1]
    fig = plt.figure(figsize=(Image.shape[0]/64,Image.shape[1]/64))
    plt.axis('off')
    plt.imshow(Image, cmap = "binary_r")
    plt.title("Assigned matching points")

    mytree = cKDTree(leftch_maxima)
    dist, indexes = mytree.query(rightch_maxima)
    for i in range(len(leftch_maxima)):
        x1, y1 = leftch_maxima[indexes[i]]
        x2, y2 = rightch_maxima[i]
        tmp_color = (ra.uniform(0, 1), ra.uniform(0, 1), ra.uniform(0, 1))
        plt.plot(x1,y1, color = tmp_color, marker = '+')
        plt.plot(x2+width,y2, color = tmp_color, marker = '+')
        plt.plot([x1,x2+width],[y1,y2], color = tmp_color)
    plt.show()

def align_by_offset(Image, shift_x, shift_y, shift_channel="right"):
    """
    shifts left or right channel to alignment.

    :param Image: 2D image array
    :param shift_x: float, channel shift in x
    :param shift_y: float, channel shift in x

    :return: 2D image array of aligned image

    :Example:

        >>> import toolbox.alignment as al
        >>> import toolbox.testdata as test
        >>> im = test.image_stack()
        >>> Dx,Dy = al.find_global_offset(im, 8,3,7)
        >>> new_image = al.align_by_offset(im[0],Dx,Dy)
    """
    left_channel = np.hsplit(Image, 2)[0]
    right_channel = np.hsplit(Image, 2)[1]
    if shift_channel == "right":
        new_coords = warp_coords(lambda xy: xy - np.array([shift_x, shift_y]), right_channel.shape)
        warped_channel = map_coordinates(right_channel, new_coords)
        aligned_image = np.concatenate((left_channel, warped_channel), axis=1)
    elif shift_channel == "left":
        new_coords = warp_coords(lambda xy: xy + np.array([shift_x, shift_y]), left_channel.shape)
        warped_channel = map_coordinates(left_channel, new_coords)
        aligned_image = np.concatenate((warped_channel, right_channel), axis=1)
    return aligned_image