from src.data_loader import NeuronLoader
import json
import os

nl = NeuronLoader()

def test_download():
	# Testing whether 'neuron_dataset' folder has been created post-download
	assert(os.path.isdir(nl.data)) == True
	# Testing whether 'neuron_dataset' folder has contents
	assert(len(os.listdir(nl.data))) != 0


def test_setup_data():
	# Testing the directory structure of 'neuron_dataset' after download
	assert os.path.isdir(os.path.join(nl.data, 'train')) == True
	assert os.path.isdir(os.path.join(nl.data, 'test')) == True 
	assert os.path.isdir(os.path.join(nl.data, 'masks')) == True
	# Testing whether the train/test/masks directories are non-empty
	assert len(os.listdir(os.path.join(nl.data, 'train'))) != 0
	assert len(os.listdir(os.path.join(nl.data, 'test'))) != 0
	assert len(os.listdir(os.path.join(nl.data, 'masks'))) != 0
	# Testing whether zip-files exist in 'neuron_dataset'
	zip_files = [zip_file for zip_file in os.listdir(nl.data) if zip_file.endswith('.zip')]	
	assert len(zip_files) == 0


def test_region_to_mask():
	# Testing the 'regions' files of all training data instances
	for train_file in nl.train_files:
		regions_path = os.path.join(nl.data, 'train', train_file, 'regions/regions.json')
		with open(regions_path, 'r') as json_file:
			regions = json.load(json_file)
			output = nl.region_to_mask(regions)
			# Testing the shape of returned mask output
			assert output.shape == (512, 512, 3)
