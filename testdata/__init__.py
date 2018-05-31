import os
from skimage.external.tifffile import imread


dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)



def single_max():
    return imread(dir_path+"/single_max.tif")

def image_stack():
    return [imread(i) for i in os.listdir("dir_path") if i.endswith(_align.tif)]