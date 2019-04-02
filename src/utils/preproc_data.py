"""						
This script explores various image pre-processing techniques to
transform the images, which can further be used as the input data
to the models.

---------------------------
Author : Caleb Crumley
"""

import numpy as np
from skimage import exposure
from skimage import external
from skimage import filters
import argparse
import os
import sys


class Preprocessing:
    def __init__(self, data, data_storage, transform, _filter):
        """
        Performs preprocessing techniques on the images dataset.

        Arguments:
        ----------
            data: str
                Directory where the training and testing data are present
            transform: str
                Transformation to perform on the image
            _filter: str
                Filtering technique to perform on the image
        """
        # Checking to make sure the neuron data directory is given correctly
        if(not os.path.isdir(data)):
            sys.exit('Data directory does not exist')
        # Directory where the new transformed data will be stored
        new_dir = os.path.join(data, data_storage)
        os.mkdir(new_dir)
        # Only taking a certain number of images from each sample
        img_iterator = 100
        #grabbing the testing and training data
        sample_dirs = [ 'train', 'test']

        for folder in sample_dirs:
            folder_path = os.path.join(data,folder)

            for sample in os.listdir(folder_path):
                # Sample should be neurfinder.00.00 and so on
                # Grabbing a list of the images' name to then preprocess
                img_dir = os.path.join(folder_path,sample)
                img_names = os.listdir(img_dir)

                i=0
                while (i < len(img_names)):
                    img = img_names[i]
                    img_path = os.path.join(img_dir,img)
                    img_array = external.tifffile.imread(img_path)

                    if (not transform == None):
                        img_array = self.transform_img(img_array, transform)

                    if (not _filter == None):
                        img_array = self.filter_img(img_array, _filter)

                    new_img_path = os.path.join(new_dir,folder,sample,img)
                    external.tifffile.imsave(new_img_path,img_array)
                    i+=img_iterator


    def transform_img(self, img, transformation):
        """
    	Applies different image preprocessing techniques

    	Arguments
    	---------
    	img : 2D numpy array
    		The image to be transformed

        transformation : str
            The specific transofromation to be applied to the img.
             Mean centering, histogram equalization, adapt histogram equalization, or a combination of them.

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


    def filter_img(self, img, _filter):
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
