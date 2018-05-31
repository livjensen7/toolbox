import skimage.filters as skim
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.optimize import curve_fit

def twoD_Gaussian(span, amplitude, mu_x, mu_y, sigma_x, sigma_y, theta, offset):
    '''
    Two dimensional Guassian function, specifically for use with point-fitting functions.

    :param span: tuple containing arrays for the range of the gaussian function in x and y.
    :param amplitude: Height of the gaussian
    :param mu_x: mean in x
    :param mu_y: mean in y
    :param sigma_x: standard deviation in x
    :param sigma_y: standard deviation in y
    :param theta: Angular offset of the coordinate axis, in radians counterclockwise from x axis.
    :param offset: Coordinate axis offset from zero

    :return: 1D array of values for the function across the range defined by span

    :Example:
        >>> import twoD_Gaussian
        >>> twoD_Gaussian()
    '''
    (x,y) = span
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-mu_x)**2) + 2*b*(x-mu_x)*(y-mu_y)
                            + c*((y-mu_y)**2)))
    return g.ravel()

def findMaxima(image,size,threshold_method = "threshold_otsu"):
    '''
    Locates point-based maxima in an image.

    :param image: 2-dimensional image array.
    :param size: int, size of the maxima and minimum filters used
    :param threshold_method: string, type of thresholding filter used. Accepts any filter in the skimmage.filters module. Default is otsu's method

    :return: 1D array of [(x,y),...] defining the locations of each maximum

    :Example:
        >>> import findMaxima
        >>> Maxima = findMaxima(Image,10)
        >>> print(Maxima)
        [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)]
    '''
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

def fitRoutine(Image, x, y, bbox):
    '''
    Fit a gaussian function to single-point based data

    :param Image: 2D array containing ROI to be fit
    :param x: center of ROI in x
    :param x: center of ROI in y
    :param bbox: length of the ROI on a side

    :return: 1D array of optimal parameters from gaussian fit.
    :return: ``None``

        if ROI falls (partially or otherwise) outside Image. Or, if curve_fit raises RuntimeError

    :Example:

        >>> import fitRoutine
        >>> fit = fitRoutine(image, x, y, 10)
        >>> print(Fit)
        [1,2,3,4,5,6]
    '''

    db = int(np.floor(bbox/2))
    span_x = np.linspace(x-db,x+db, bbox)
    span_y = np.linspace(y-db,y+db, bbox)
    X,Y = np.meshgrid(span_x, span_y)
    if 0<= y-db <= y+db+1 <= Image.shape[0] and 0<= x-db <= x+db+1 <= Image.shape[1]:
        pixel_vals = Image[y-db:y+db+1, x-db:x+db+1].ravel()
        scaled = [k/max(pixel_vals) for k in pixel_vals]
        initial_guess = (1, x, y, 1, 1, 0, 0)
        try:
            popt, pcov = curve_fit(function, (X, Y), scaled, p0=initial_guess)
        except RuntimeError:
            popt = None
    else:
        popt = None
    return popt