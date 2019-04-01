import os
import json
import thunder as td
from extraction import NMF

def nmf():
    """
    Uses the NMF algorithm to segment out regions and store coordinates in
    a JSON file
    Arguments
    ----------
    datasets: list
        A list of all the names of folders that contain images
    """
    datasets = [
  '00.00.test', '00.01.test', '01.00.test', '01.01.test', '02.00.test', '02.01.test', '03.00.test', '04.00.test', '04.01.test']

    submission = []
    for dataset in datasets:
        print('Loading dataset: %s ' %dataset)
	dataset_path = 'neurofinder.' + dataset
        path = os.path.join('/home/aashish_yadavally1995/neuron_dataset/test', dataset_path, 'images')
        # Getting the images data from path
        data = td.images.fromtif(path, ext='tiff')
        nmf = NMF(k=10, percentile=99, max_iter=50, overlap=0.1)
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

nmf()
