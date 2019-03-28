import os
import subprocess
import zipfile
import shutil
import numpy as np
import json
import cv2
<<<<<<< HEAD
import matplotlib.pyplot as plt
from scipy.misc import imread
from glob import glob
=======
from regional import many
>>>>>>> a8f602fadf36f03791345f0ade428d1bae096b9e


def region_to_mask(file):
	"""
	Converts the regions JSON file into mask

	Arguments
	---------
	path : string
		Path to JSON file which needs to be converted into corresponding mask

	Returns
	-------
	output : 2D numpy array
		Mask image
	"""
	files = sorted(glob(file'/images/*.tiff'))
	imgs = array([imread(f) for f in files])
	print(imgs.shape)
    	dims = imgs.shape[1:]

	"""
	load the regions (training data only)
	"""
	with open(file'/regions/regions.json') as f:
    		regions = json.load(f)

	def tomask(coords):
	    mask = zeros(dims)
	    c,v = zip(*coords)
	    mask[c,v] = 1
	    return mask

	masks = array([tomask(s['coordinates']) for s in regions])
	mean_masks = np.mean(masks, axis=0)
	imgs = np.reshape(imgs, imgs.shape+(1,))
	mean_masks = mean_masks[...,np.newaxis]
	mean_masks = mean_masks[np.newaxis,...]

	dl = DataLoader()
	dl.region_to_mask(file)


def mask_to_region(image):
	"""
	Converts mask image into corresponding regions JSON file

	Arguments
	---------
	image : 2D numpy array
		Mask image

	Returns
	-------
	output : list
		List to be written into JSON file
	"""
	nl = NeuronLoader()
	nl.mask_to_region(image)


class NeuronLoader:
	"""
	1. Downloads 'NeuronFinder'	dataset from Google Storage Bucket
	2. Sets up data folder with 'train', 'test' subdirectories
	3. Converts 'regions'  into masks and vice-versa
	4. Saves masks of 'train' files in 'masks' subdirectory
	"""
	def __init__(self, gs_url, data, train_opts, test_opts):
		"""
		Initializes NeuronLoader class

		Arguments
		---------
		gs_url : str
			Google Storage Bucket link from which dataset shall be downloaded
		data : str
			Name of dataset folder
		train_opts : list
			List of train files to download
		test_opts : list
			List of test files to download
		"""
		self.url = gs_url
		self.data = data
		self.train_opts = train_opts
		self.test_opts = test_opts
		self.train_files = ['neurofinder.' + train_opt  for train_opt in self.train_opts]
		self.test_files = ['neurofinder.' + test_opt  for test_opt in self.test_opts]
		if os.path.isdir(self.data):
			if 'train' in os.listdir(self.data):
				self.setup_data()
		else:
			self.download()
			self.setup_data()


	def download(self):
		"""
		Downloads all zip files from Google Storage Bucket into data directory in
		Google Cloud Platform VM Instance
		"""
		subprocess.call('mkdir ' + self.data, shell=True)
		# Downloading train/test files as per user's choice
		zip_train_files = [train_file + '.zip' for train_file in self.train_files]
		for zip_train_file in zip_train_files:
			subprocess.call('/usr/bin/gsutil -m rsync -r ' +
				os.path.join(self.url, zip_train_file) + ' ' + self.data, shell=True)
		zip_test_files = [test_file + '.test.zip' for test_file in self.test_files]
		for zip_test_file in zip_test_files:
			subprocess.call('/usr/bin/gsutil -m rsync -r ' +
				os.path.join(self.url, zip_test_file) + ' ' + self.data, shell=True)

	def setup_data(self):
		"""
		Sets up data folder with 'train', 'test' subdirectories; converts 'regions'
		into masks and vice-versa; saves masks of 'train' files in 'masks' subdirectory
		"""
		path = self.data
		zip_files = [zip_file for zip_file in os.listdir(path) if zip_file.endswith('.zip')]
		# Unzipping all train/test data directories
		for zip_file in zip_files:
			zip_ref = zipfile.ZipFile(os.path.join(path, zip_file), 'r')
			zip_ref.extractall(path)
			zip_ref.close()
			# Removing zip files from data directory
			shutil.rmtree(os.path.join(path, zip_file))
		# Move all train files into 'train' folder
		os.mkdir(os.path.join(self.data, 'train'))
		for train_file in self.train_files:
			shutil.move(os.path.join(self.data, train_file),
				os.path.join(self.data, 'train'))
		# Move all test files into 'test' folder
		os.mkdir(os.path.join(self.data, 'test'))
		for test_file in self.test_files:
			shutil.move(os.path.join(self.data, test_file),
				os.path.join(self.data, 'test'))
		# Convert all regions into masks and save in 'masks' folder
		os.mkdir(os.path.join(self.data, 'masks'))
		for train_file in self.train_files:
			regions_path = os.path.join(self.data, train_file, 'regions/regions.json')
			with open(regions_path, 'r') as json_file:
				regions = json.load(json_file)
				output = self.region_to_mask(regions)
				cv2.imwrite(os.path.join(self.data, 'masks', train_file + '.png'), output)

<<<<<<< HEAD

	def region_to_mask(self, file):
	"""
	Converts region file into mask
=======

	def region_to_mask(self, path):
		"""
		Converts the regions JSON file into mask
>>>>>>> a8f602fadf36f03791345f0ade428d1bae096b9e

		Arguments
		---------
		path : string
			Path to JSON file which needs to be converted into corresponding mask

		Returns
		-------
		output : 2D numpy array
			Mask image
		"""
		with open(path, 'r') as f:
			regions_json = json.load(f)
		regions = many([region['coordinates'] for region in regions_json])
		_mask = truth.mask(dims=(512,512), stroke='white', fill='white', background='black')
		return _mask


	def mask_to_region(self, image):
		"""
		Converts mask image into corresponding regions list

		Arguments
		---------
		image : 2D numpy array
			Mask image

<<<<<<< HEAD
	Returns
	-------
	output : list
		List to be written into JSON file
	"""
=======
		Returns
		-------
		output : list
			List to be written into JSON file
		"""
>>>>>>> a8f602fadf36f03791345f0ade428d1bae096b9e
