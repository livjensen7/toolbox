import os
from skimage.external.tifffile import imread


dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)



def single_max():
    return imread(dir_path+"/single_max.tif")