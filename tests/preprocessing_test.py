import numpy as np
from skimage import exposure
from skimage import external
from skimage import filters

'''
This file takes the two functions in utils/preproc_data.py and runs test on
them to assure that they are returning the proper output's type, size and
'''


''' the functions to test'''
def transform_img(img, transformation):
    """
	Applies different image preprocessing techniques

	Arguments
	---------
	img : 2D numpy array
		The image to be transformed

    transformation : str
        The specific transofromation to be applied to the img.
        Mean centering, histogram equalization, adapt histogram equalization,
        or a combination of them.

	Returns
	-------
	output : 2D numpy array
		The transformed image
	"""
    if transformation == 'mean':
        img = img.mean - img

    if transformation == 'hist':
        img = exposure.equalize_hist(img)

    if transformation == 'adapthist':
        img = exposure.equalize_adapthist(img)

    if  transformation == 'mean_hist':
        img = img.mean - img
        img = exposure.equalize_hist(img)

    if transformation == 'mean_adapt':
        img = img.mean - img
        img = exposure.equalize_adapthist(img)
    return img


def filter_img(img, _filter):
    """
	Applies different image preprocessing filters

	Arguments
	---------
	img : 2D numpy array
		The image to be transformed

    _filter : str
        The specific transofromation to be applied to the img. Gaussian and Median

	Returns
	-------
	output : 2D numpy array
		The filtered image
	"""
    if _filter == 'gaus':
        img = filters.gaussian(img)

    if _filter == 'median':
        img = filters.median(img)
    return img



'''testing the functions'''
def test_transform_img():
    arr = np.random.normal(loc=0, scale=3, size=(10, 5))

    transformations = ['mean','hist','adapthist','mean_hist','mean_adapt']
    for tansform in transformations:
        test_arr = transform_img(arr,transform)
        assert instance(test_arr,arr)
        assert size(test_arr) == size(arr)


def test_filter_img():
    arr = np.random.normal(loc=0, scale=3, size=(10, 5))

    filters = ['median','gaus']
    for filter in filters:
        test_arr = filter_img(arr,filter)
        assert instance(test_arr,arr)
        assert size(test_arr) == size(arr)
