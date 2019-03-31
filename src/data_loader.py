import os
import subprocess
import zipfile
import shutil
import numpy as np
import json
import cv2
from regional import many
import site
site.addsitedir(os.path.dirname(os.path.realpath(__file__)))

def region_to_mask(path):
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
	nl = NeuronLoader()
	nl.region_to_mask(path)

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
	def __init__(self, gs_url='gs://uga-dsp/project3', data='neuron_dataset', 
		train_opts=['00.00', '00.01', '00.02', '00.03', '00.04', '00.05', '00.06',
		'00.07', '00.08', '00.09', '00.10', '00.11', '01.00', '01.01', '02.00',
		'02.01', '03.00', '04.00', '04.01'], test_opts=['00.00', '00.01', '01.00', 
		'01.01', '02.00', '02.01', '03.00', '04.00', '04.01']):
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
		self.test_files = ['neurofinder.' + test_opt + '.test' for test_opt in self.test_opts]
		if os.path.isdir(self.data):
			print("NeuroFinder Dataset has already been downloaded...")
			print("Setting up 'neuron_dataset' folder...")
			self.setup_data()
		else:
			print('Downloading NeuroFinder Dataset...')
			self.download()
			print("Setting up 'neuron_dataset' folder...")
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
			subprocess.call('/usr/bin/gsutil -m cp -r ' +
				os.path.join(self.url, zip_train_file) + ' ' + self.data, shell=True)
		zip_test_files = [test_file + '.zip' for test_file in self.test_files]
		print(zip_test_files)
		for zip_test_file in zip_test_files:
			subprocess.call('/usr/bin/gsutil -m cp -r ' +
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
			print('Unziping ' + zip_file + '...')
			zip_ref = zipfile.ZipFile(os.path.join(path, zip_file)).extractall(path)
			# Removing zip files from data directory
			os.remove(os.path.join(path, zip_file))
		# Move all train files into 'train' folder
		if 'train' in os.listdir(self.data): # If 'train' folder exists
			if len(os.listdir(os.path.join(self.data, 'train'))) == 0: # If 'train' folder is empty
				for train_file in self.train_files:
					shutil.move(os.path.join(self.data, train_file), 
						os.path.join(self.data, 'train'))
				print("'train' folder has been successfully created!")
			else:
				print("'train' folder already exists. Moving ahead...")
		else: # Creating 'train' folder
			os.mkdir(os.path.join(self.data, 'train'))
			for train_file in self.train_files:
				shutil.move(os.path.join(self.data, train_file), 
					os.path.join(self.data, 'train'))
			print("'train' folder has been successfully created!")

		# Move all test files into 'test' folder
		if 'test' in os.listdir(self.data): # If 'test' folder exists
			if len(os.listdir(os.path.join(self.data, 'test'))) == 0: # If 'test' folder is empty
				for test_file in self.test_files:
					shutil.move(os.path.join(self.data, test_file), 
						os.path.join(self.data, 'test'))
				print("'test' folder has been successfully created!")
			else:
				print("'test' folder already exists. Moving ahead...")
		else: # Creating 'test' folder
			os.mkdir(os.path.join(self.data, 'test'))
			for test_file in self.test_files:
				shutil.move(os.path.join(self.data, test_file), 
					os.path.join(self.data, 'test'))
			print("'test' folder has been successfully created!")

		# Convert all regions into masks and save in 'masks' folder
		if 'masks' in os.listdir(self.data):
			if len(os.listdir(os.path.join(self.data, 'masks'))) == 0:
				for train_file in self.train_files:
					regions_path = os.path.join(self.data, 'train', train_file, 'regions/regions.json')
					with open(regions_path, 'r') as json_file:
						regions = json.load(json_file)
						output = self.region_to_mask(regions)
						cv2.imwrite(os.path.join(self.data, 'masks', train_file + '.png'), output)
				print("'masks' folder has been successfully created!")			
			else:
				print("'masks' folder already exists. Moving ahead...")
		else:
			os.mkdir(os.path.join(self.data, 'masks'))
			for train_file in self.train_files:
				regions_path = os.path.join(self.data, 'train', train_file, 'regions/regions.json')
				with open(regions_path, 'r') as json_file:
					regions = json.load(json_file)
					output = self.region_to_mask(regions)
					cv2.imwrite(os.path.join(self.data, 'masks', train_file + '.png'), output)
			print("'masks' folder has been successfully created!")


	def region_to_mask(self, regions_json):
		"""
		Converts region JSON file into mask

		Arguments
		---------
		regions_json : json file
			JSON file which needs to be converted into corresponding mask

		Returns
		-------
		output : 2D numpy array
			Mask image
		"""
		regions = many([region['coordinates'] for region in regions_json])
		_mask = regions.mask(dims=(512,512), stroke='white', fill='white', background='black')
		
		return _mask


	def mask_to_region(self, image):
		"""
		Converts mask image into corresponding regions list

		Arguments
		---------
		image : 2D numpy array
			Mask image

		Returns
		-------
		output : list
			List to be written into JSON file
		"""
