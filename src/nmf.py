import os
import json
import thunder as td
from extraction import NMF

def nmf(datasets):
    """
    Uses the NMF algorithm to segment out regions and store coordinates in
    a JSON file
    Arguments
    ----------
    datasets: list
        A list of all the names of folders that contain images
    """
    submission = []
    for dataset in datasets:
        print('Loading dataset: %s ' %dataset)
        path = os.path.join('neuron_dataset/test', dataset, 'images')
        # Getting the images data from path
        data = td.images.fromtif(path, ext='tiff')
        nmf = NMF(k=5, percentile=99, max_iter=50, overlap=0.1)
        # Fitting on the given dataset
        model = nmf.fit(data, chunk_size=(50,50), padding=(25,25))
        merged = model.merge(0.1)
        # Storing found regions
        regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
        result = {'dataset': dataset, 'regions': regions}
        submission.append(result)
    # Writing the results to submission.json
    with open('submission.json', 'w') as f:
        f.write(json.dumps(submission))
