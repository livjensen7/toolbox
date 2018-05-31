from skimage.external.tifffile import imread
import os


test_maximum = imread("single_max.tif")
test_image = imread("1_align.tif")
test_images = [imread(i) for i in os.listdir('.') if i.endswith("_align.tif")]
