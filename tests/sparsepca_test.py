
from src.sparsepca import sparse_pca
from src.data_loader import NeuronLoader
import os

nl = NeuronLoader()

def sparse_pca_test():
    data_folder = 'test_folder'
    sparse_pca(nl.test_files, data_folder, test=True)
    #Testing if submissions.json file was created
    assert os.path.isfile('submission.json')
    #Now computing test to assure certain aspects within the file
	with open('submission.json', 'r'):
		regions = json.load(json_file)
	# NMF is run on the Test Set
	# There are 9 testing videos in the dataset
	assert len(regions) <= 9
	for _set in regions:
		# Testing that the dataset belongs to test set
		assert _set["dataset"] in nl.test_files
		num_coordinates = 0
		for i in range(len(_set["regions"])):
			num_coordinates += len(_set["regions"][i]["coordinates"])
		# Testing that the number of regions' coordinates is less than
		# image dimensions
		assert num_coordinates < 512*512
