import numpy as np
from skimage import exposure
from skimage import filters

def mean_center(img):

    return img.mean - img

def hist_equalization(img):

    return exposure.equalize_hist(img)

def adapthist_equalization(img):

    return exposure.equalize_adapthistIimg)

def mean_hist(img):

    mean = img.mean - img

    return exposure.equalize_hist(mean)

def gaussian_filter(img):

    return filters.gaussian(img)

def median_filter(img):

    return filters.median(img)



