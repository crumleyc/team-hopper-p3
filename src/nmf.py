import json
import thunder as td
from extraction import NMF

def nmf(data):
        """
        Uses the nmf algorithm to segment out regions and store coordinates in
        a JSON file
        Arguments
        ----------
        data: list
            A list of all the names of folders that contain images
        """



        submission = []

        for dataset in datasets:
          #Getting the data from the path
          print('loading dataset: %s' % dataset)
          path = 'neurofinder.' + dataset
          data = td.images.fromtif(path + '/images', ext='tiff')

          nmf = NMF(k=5, percentile=99, max_iter=50, overlap=0.1)
          #Fitting on the given dataset
          model = nmf.fit(data, chunk_size=(50,50), padding=(25,25))

          merged = model.merge(0.1)
          #Storing found regions
          regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
          result = {'dataset': dataset, 'regions': regions}
          submission.append(result)

        #Writing the results to submission.json
        with open('submission.json', 'w') as f:
          f.write(json.dumps(submission))
