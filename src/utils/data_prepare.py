import json
import matplotlib.pyplot as plt
from numpy import array, zeros, reshape
import numpy as np
from scipy.misc import imread
from glob import glob

#Loading the images

files = sorted(glob('~/neuron_dataset/train/neurofinder.*/images/*00.tiff'))
imgs = np.array([imread(f) for f in files])
print(imgs.shape)


dims = imgs.shape[1:]


def tomask(coords):
    """
    To assign 1 to the region coordinates to convert to masks
    Arguments
    ---------
    coords: list
        A list of coordinates of a particular regions

    Returns
    --------
    mask: list
    """
    mask = zeros(dims)
    c,v = zip(*coords)
    mask[c,v] = 1
    return mask

def prepare(file):
    """
    To prepare numpy arrays of masks
    Arguments
    ---------
    file: string
        Path to regions.json

    Returns
    --------
    _mask: numpy array
    """
    with open(file) as f:
            regions = json.load(f)

    masks = array([tomask(s['coordinates']) for s in regions])

    mean_masks = np.mean(masks, axis=0)
    #mean_masks = mean_masks[...,np.newaxis]
    _mask = np.empty((0,512,512))
    print(len(regions))
    for i in range(0,31):
        _mask=np.append(_mask,[mean_masks], axis = 0)


    return _mask

region_files = sorted(glob('~/neuron_dataset/train/neurofinder.*/regions/regions.json'))
final_masks = np.empty((0,512,512))

#Appending all the mask numpy arrays together
for file in region_files:
    _mask = prepare(file)
    final_masks = np.append(final_masks, _mask, axis = 0)

#Adding the channel dimension
imgs = np.reshape(imgs, imgs.shape+(1,))
print(imgs.shape)

final_masks = final_masks[...,np.newaxis]
print(final_masks.shape)
