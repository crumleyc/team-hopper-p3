"""                     
This script performs NMF matrix factorization technique on the test datasets 
passed as parameter to the `nmf` function. The code is based on thunder-extraction 
NMF algorithm.

    References:
    -----------
    1) https://github.com/thunder-project/thunder-extraction/blob/master/example.ipynb

---------------------------
Author : Rutu Gandhi
"""


import os
import json
import thunder as td
from extraction import NMF

class Nmf:
    """
    Uses the NMF algorithm to segment out regions and store coordinates 
    of regions in the JSON file.
    """
    def __init__(self, datasets, k=5, percentile=99, max_iter=50, merge_ratio=0.1, test=False):
        """
        Initializes the Nmf class for finding regions of neurons

        Arguments
        ----------
        datasets: list
            A list of all the names of folders that contain images
        k : int
            Number of components in NMF
        percentile: int
            Threshold above which regions will be selected
        max_iter: int
            Maximum number of iterations NMF algorithm will run for
        merge_ratio: float
            Overlap between two adjacent neurons while mapping regions
        test: boolean
            Value indicates whether test suite is being run, or model in general
        """
        self.datasets = datasets
        self.k = k
        self.percentile = percentile
        self.max_iter = max_iter
        self.merge_ratio = merge_ratio
        self.test = test


    def get_output(self):
        """
        Writes output of NMF model into JSON file with the name
        `submission.json`
        """
        submission = []
        for dataset in self.datasets:
            print('Loading dataset: %s ' %dataset)
            dataset_path = 'neurofinder.' + dataset
            path = os.path.join('D:/Spring2019/DataSciencePracticum/p3', dataset_path, 'images')
            
            # Getting the images data from path
            # Returns only first image of dataset if test is True
            if self.test == True:
                data = td.images.fromtif(path, ext='tiff').first()
            else:
                data = td.images.fromtif(path, ext='tiff')

            nmf = NMF(k=self.k, percentile=self.percentile, max_iter=self.max_iter, 
                overlap=0.1)
            
            # Fitting on the given dataset
            model = nmf.fit(data, chunk_size=(100,100), padding=(25,25))
            merged = model.merge(self.merge_ratio)
            
            # Storing found regions in the required format
            regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
            result = {'dataset': dataset, 'regions': regions}
            submission.append(result)
        # Writing the results to submission.json
        with open('submission.json', 'w') as f:
            f.write(json.dumps(submission))
