import json
import thunder as td
from extraction import NMF

#list of the test dataset
datasets = [
  '00.00.test', '00.01.test', '01.00.test', '01.01.test', '02.00.test', '
]

"""list to store the dictionary (key->dataset name,
value-> list of region coordinates"""
submission = []

for dataset in datasets:
  print('loading dataset: %s' % dataset)
  path = 'neurofinder.' + dataset
  data = td.images.fromtif(path + '/images', ext='tiff')
  print('finding regions')
  nmf = NMF(k=5, percentile=99, max_iter=50, overlap=0.1)
  model = nmf.fit(data, chunk_size=(50,50), padding=(25,25))
  merged = model.merge(0.1)
  print('found %g regions' % merged.regions.count)
  regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
  result = {'dataset': dataset, 'regions': regions}
  submission.append(result)

print('writing results')
with open('submission.json', 'w') as f:
  f.write(json.dumps(submission))
