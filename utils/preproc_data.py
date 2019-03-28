import numpy as np
from skimage import exposure
from skimage import external
from skimage import filters

import argparse
import os 
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--transform',type=str,help='Select the image transformation',
                    choices=['mean','hist','adapthist','mean_hist','mean_adapt'])
parser.add_argument('--filter',type=str,help='Select the filter to reduce the noise in the imgages',
                    choices=['gaus','median'])
args = parser.parse_args()


#The directory where the training and testing data are
data = 'neuron_data'

#Checking to make sure the neuron data directory is given correctly
if(not os.path.isdir(data)):
    sys.exit('Data directory does not exist')


#where the new transformed data will be stored
new_dir = 'preprec_neuron_data'
os.mkdir(new_dir)

#Only taking a certain number of images from each sample
img_iterator = 100



for dir in os.listdir(data):
    # dir should be train or test
    
    sample_dirs = os.path.join(data,dir)

    
    for sample in os.listdir(sample_dirs):
        #sample should be neurfinder.00.00 and so on

        #Grabbing a list of the images' name to then preprocess
        img_dir = os.path.join(sample_dirs,sample)
        img_names = os.listdir(img_dir)

        i=0
        while (i < len(img_names)):
            
            img = img_names[i]
            img_path = os.path.join(img_dir,img)

            img_array = external.tifffile.imread(img_path)

            if (not args.transform == None):
                
                img_array = transform_img(img_array,args.transform)

            if (not args.filter == None):

                img_array = filter_img(img_array, args.filter)

            new_img_path = os.path.join(new_dir,dir,sample,img)
            external.tifffile.imsave(new_img_path,img_array)

            
            i+=img_iterator


def transform_img(img,transformation):
    """
	Applies different image preprocessing techniques

	Arguments
	---------
	img : 2D numpy array
		The image to be transformed

    transformation : str
        The specific transofromation to be applied to the img. Mean centering, histogram equalization, adapt histogram equalization, or a combination of them.

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

def filter_img(img, filter):
    """
	Applies different image preprocessing filters

	Arguments
	---------
	img : 2D numpy array
		The image to be transformed

    filter : str
        The specific transofromation to be applied to the img. Gaussian and Median

	Returns
	-------
	output : 2D numpy array 
		The filtered image
	"""
    
    if filter == 'gaus':
        img = filters.gaussian(img)
    
    if filter == 'median':
        img = filters.median(img)

    return img



