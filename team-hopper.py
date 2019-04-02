import argparse
import subprocess
from src.data_loader import NeuronLoader
from src.utils.preproc_data import Preprocessing
from src.nmf import nmf


parser = argparse.ArgumentParser(description='Team Hopper : Neuron Finder')
parser.add_argument('--model', dest='model', type=str, choices=['nmf', 'unet'],
	default='nmf', help='model to find neurons')
parser.add_argument('--url', dest='url',type=str, default='gs://uga-dsp/project3',
	help='google storage bucket link')
parser.add_argument('--data', dest='data', type=str, default='neuron_dataset',
	help='name of folder to download neuron dataset into')
parser.add_argument('--preprocess', dest = 'preprocess', type=bool, default=False,
	choices=[True, False], help='choice to work with preprocessed data in unet')
parser.add_argument('--train_opts', dest='train_opts', type=str, default='all',
	choices=['00.00', '00.01', '00.02', '00.03', '00.04', '00.05', '00.06',
	'00.07', '00.08', '00.09', '00.10', '00.11', '01.00', '01.01', '02.00',
	'02.01', '03.00', '04.00', '04.01', 'all'],
	help='training data to be downloaded - part/all')
parser.add_argument('--test_opts', dest='test_opts', type=str, default='all',
	choices=['00.00', '00.01', '01.00', '01.01', '02.00', '02.01', '03.00',
	'04.00', '04.01', 'all'], help='test data to be downloaded - part/all')
parser.add_argument('--transform', dest = 'transform', type=str, default='None',
	choices=['mean', 'hist', 'adapthist', 'mean_hist', 'mean_adapt', 'None'],
	help='Select the image transformation')
parser.add_argument('--filter', dest='filter', type=str, default='None',
	choices = ['gaus', 'median', 'None'])
parser.add_argument('--test', dest='test', type=bool, default=False,
	choices=[True, False], help='to run the test suite')

args = parser.parse_args()
url = args.url
model = args.model
data = args.data
preprocess = args.preprocess
transform = args.transform
_filter = args.filter
test = args.test

if args.train_opts != 'all':
	train_opts = args.train_opts.split()
else:
	train_opts = ['00.00', '00.01', '00.02', '00.03', '00.04', '00.05', '00.06',
	'00.07', '00.08', '00.09', '00.10', '00.11', '01.00', '01.01', '02.00',
	'02.01', '03.00', '04.00', '04.01']

if args.test_opts != 'all':
	test_opts = args.test_opts.split()
else:
	test_opts = ['00.00', '00.01', '01.00', '01.01', '02.00', '02.01', '03.00',
	'04.00', '04.01']

nl = NeuronLoader(url, data, train_opts, test_opts)

if test:
	subprocess.call('python -m pytest', shell=True)
else:
	if model == 'nmf':
		if preprocess :
			print('Preprocessing techniques have not been tested for NMF.')
			print('Proceeding with regular NMF technique.')
			nmf(nl.test_files)
		else:
			nmf(nl.test_files)
	elif model == 'unet':
		if preprocess :
			'''Will get an error unless you add the name of the subdirectory'''
			pp = Preprocessing(nl.data, transform, _filter)
			# Execute UNet here
			pass
		else:
			# Execute UNet here
			pass
