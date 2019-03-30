from ..nmf import nmf
from ..data_loader import NeuronLoader
import json


nl = NeuronLoader()


def test_nmf():
	# Running NMF algorithm on all test sets
	nmf(nl.test_files)
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
