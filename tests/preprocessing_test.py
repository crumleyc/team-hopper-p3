import numpy as np
from skimage import exposure
from skimage import external
from skimage import filters

'''
This file takes the two functions in utils/preproc_data.py and runs test on
them to assure that they are returning the proper output's type, size and
'''


''' the funtion to test'''
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

'''testing the function'''
def test_transform_img():
    arr = np.random.normal(loc=0, scale=3, size=(10, 5))

    transformations = ['mean','hist','adapthist','mean_hist','mean_adapt']
    for tansform in transformations:

        test_arr = transform_img(arr,transform)
        assert instance(test_arr,arr)
        assert size(test_arr) == size(arr)
