from skimage.external.tifffile import imread
import os, sys
sys.path.insert(0, os.path.abspath(".."))

print(os.listdir("."))
test_maximum = imread("testdata/single_max.tif")
test_image = imread("1_align.tif")
#test_images = [imread(i) for i in os.listdir('.') if i.endswith("_align.tif")]
