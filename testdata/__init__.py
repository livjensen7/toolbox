from skimage.external.tifffile import imread
pa =testdata.__path__

print(os.listdir(pa))
test_maximum = imread("testdata/single_max.tif")
test_image = imread("1_align.tif")
#test_images = [imread(i) for i in os.listdir('.') if i.endswith("_align.tif")]
