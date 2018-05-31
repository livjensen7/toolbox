import os
import skimage.external.tifffile as ski


dir_path = os.path.dirname(os.path.realpath(__file__))

def single_max():
    return ski.imread(dir_path+"/single_max.tif")

def image_stack():
    return [ski.imread(dir_path+i) for i in os.listdir(dir_path) if i.endswith("_align.tif")]