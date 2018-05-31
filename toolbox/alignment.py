import numpy as np
from toolbox.point_fitting import twoD_Gaussian, findMaxima, fitRoutine

def FD_rule_bins(data):
    '''
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
    '''
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    opt_binwidth = 2*iqr/(len(data)**(1/3.))
    return np.arange(min(data), max(data) + opt_binwidth, opt_binwidth)


def scrub_outliers(data,recurse = 2):
    '''
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
    '''

    recur_count = 0
    while recur_count<recurse:
        recur_count += 1
        data = [datum for datum in data if
                datum < np.mean(data) + 2*np.std(data) and
                datum > np.mean(data) - 2*np.std(data)]
    return data





def get_offset_distribution(Image,dx,dy,bbox):
    '''
    Image passed to this function should be 2-channel data divided vertically.
    This function in order:
        * splits the image into left and right channels
        * locates and fits all of the foci in each channel
        * pairs up associated foci from each channel and determines their x- and y- offset

    :param Image: 2D image array
    :param dx: int, maximum distance in x to look for corresponding points in separate channels.
    :param dy: int, maximum distance in y to look for corresponding points in separate channels.
    :param bbox: int, size of ROI around each point to apply gaussian fit.

    :return: Two lists containing the x- and y- offsets of each corresponding pair of foci.

    :Example:

        >>> from toolbox.alignment import get_offset_distribution
        >>> import toolbox.testdata as test
        >>> im = test.image_stack()[0]
        >>> A,B = get_offset_distribution(im, 8,3,8)




    '''
    leftch_maxima = findMaxima(np.hsplit(Image, 2)[0],10)
    rightch_maxima = findMaxima(np.hsplit(Image, 2)[1],10)
    Delta_x,Delta_y = [],[]
    for x1,y1 in leftch_maxima:
        for x2,y2 in rightch_maxima:
            if abs(x1-x2)<=dx and abs(y1-y2)<=dy:
                fit_ch1 = fitRoutine(twoD_Gaussian, np.hsplit(Image, 2)[0], x1, y1, bbox)
                fit_ch2 = fitRoutine(twoD_Gaussian, np.hsplit(Image, 2)[1], x2, y2, bbox)
                try:
                    Delta_x.append(fit_ch1[1]-fit_ch2[1])
                    Delta_y.append(fit_ch1[2]-fit_ch2[2])
            
                except TypeError:
                    pass
    return(Delta_x,Delta_y)

def findGlobalOffset(file_list, dx, dy, bbox):
    '''
    finds the optimal x-shift and y-shift of the data.

    :param file_list: 1D list of file paths for images used in determination of the offset
    :param dx: int, maximum distance in x to look for corresponding points in separate channels.
    :param dy: int, maximum distance in y to look for corresponding points in separate channels.
    :param bbox: int, size of ROI around each point to apply gaussian fit.

    :return: Mean x and y shift values to align all images best fit.
    '''
    pooled_x, pooled_y = [], []
    for fname in file_list:
        im = imread(fname)
        xdist, ydist = getOffsetDistribution(im, dx, dy, bbox)
        pooled_x += scrubOutliers_recursive(xdist)
        pooled_y += scrubOutliers_recursive(ydist)
    mu1, sigma1 = norm.fit(pooled_x)
    mu2, sigma2 = norm.fit(pooled_y)
    return mu1, mu2