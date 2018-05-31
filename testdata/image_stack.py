from skimage.external.tifffile import imread
path_to_image =testdata.__path__

print(path_to_image)
test_maximum = imread(path_to_image+"/single_max.tif")
