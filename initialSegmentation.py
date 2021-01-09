#!/usr/bin/env python3

import json
import numpy as np
import nibabel as nib
from matplotlib import cm
from scipy.io import loadmat, savemat
from dipy.segment.clustering import QuickBundles
from dipy.tracking.streamline import length
from dipy.tracking.metrics import mean_curvature
from dipy.io.streamline import load_tractogram

# initial quickbundles segment into 3 for testing :D
def quickClean(streamlines,number_bundles):

	# set up counter for n_clusters and start threshold distance at 1mm.
	n_clusters = 0
	thrs = 1
	
	# loop through clustering until number_bundles of bundles segmented
	while n_clusters != number_bundles:
		qb = QuickBundles(threshold=thrs)
		clusters = qb.cluster(streamlines)
		n_clusters = len(clusters)
		thrs=thrs+1
		if n_clusters == (number_bundles-1):
			thrs=thrs-2
			for i in np.linspace(thrs,thrs+1,num=1000):
				print(i)
				qb = QuickBundles(threshold=i)
				clusters = qb.cluster(streamlines)
				n_clusters = len(clusters)
				if n_clusters == number_bundles:
					break

	print(thrs)

	return clusters

# make classification structure
def makeClassificationStructure(tract_indices,streamlines,clusterDict,Names,index_labels):

	## start classification creation
	# make names array and empty streamline endex for classification structure
	names = np.array(Names,dtype=object)
	streamline_index = np.zeros(len(streamlines))
	tractsfile = []

	# loop through bundles
	for bnames in range(np.size(names)):
		# extract the bundle segment number following quickbundles segmentation
		segment_number = int(names[bnames].split('_')[::-1][0])
		
		# identify the index into clusterDict 
		if segment_number == 1:
			names_index = int((Names.index(names[bnames]) / 3) + 1)
		
		# identify cluster indices
		tract_ind = clusterDict[names_index][segment_number-1].indices

		# loop through cluster indices and make it's value in the streamline_index equal to the inputted index_labels
		for ss in range(len(tract_ind)):
			streamline_index[tract_indices[names_index][tract_ind[ss]]] = int(index_labels[bnames])
	    
	    ## starting json creation
	    # extract bundle streamlines and make empty array of length of bundle streamlines
		bundle_streamlines = streamlines[streamline_index == index_labels[bnames]]
		jsonStreamlines = np.zeros([len(bundle_streamlines)],dtype=object)

		# loop through streamlines, transpose the order of the matrix, and round values by 2 for visualization
		for e in range(len(jsonStreamlines)):
		    jsonStreamlines[e] = np.transpose(bundle_streamlines[e]).round(2)

		# create color for each tract
		color=list(cm.nipy_spectral(bnames*10))[0:3]
		
		# identify count of bundle
		count = len(bundle_streamlines)

		# reshape for json structure and create coords field for json structure
		jsonfibers = np.reshape(jsonStreamlines[:count], [count,1]).tolist()
		for i in range(count):
		    jsonfibers[i] = [jsonfibers[i][0].tolist()]

		# write out json structure
		with open ('wmc/tracts/'+str(bnames+1)+'.json', 'w') as outfile:
		    jsonfile = {'name': names[bnames], 'color': color, 'coords': jsonfibers}
		    json.dump(jsonfile, outfile)

		# append information to big json file
		tractsfile.append({"name": names[bnames], "color": color, "filename": str(bnames+1)+'.json'})

	# output big json file once loop is done
	with open ('wmc/tracts/tracts.json', 'w') as outfile:
	    json.dump(tractsfile, outfile, separators=(',', ': '), indent=4)

	# save classification structure
	print("saving classification.mat")
	savemat('wmc/classification.mat', { "classification": {"names": names, "index": streamline_index }})

def main():
	
	# load config.json structure
	with open('config.json','r') as config_f:
		config = json.load(config_f)

	# parse inputs
	number_bundles = int(config['number_bundles'])

	# load data
	reference = nib.load(config['dwi'])
	sft = load_tractogram(config['tractogram'],reference)
	classification = loadmat(config['classification'],squeeze_me=True)['classification']

	# extract tract names and indices
	tract_names = classification['names'].tolist().tolist()
	indices = classification['index'].tolist()
	tract_index_labels = np.unique(indices)
	tract_indices = {}
	for ti in tract_index_labels:
		tract_indices[ti] = [ i for i in range(len(indices)) if indices[i] == ti ]

	# segment clusters with quickbundles
	clusterDict = {}
	Names = []
	for tt in range(len(tract_index_labels)):
		print("%s" %tract_names[tt])
		clusterDict[tract_index_labels[tt]] = quickClean(sft.streamlines[tract_indices[tract_index_labels[tt]]],number_bundles)

		# write out names for classification generation
		Names = Names + [ "%s_%s" %(tract_names[tt],f) for f in range(1,4)  ]

	# index labels to use in classification structure
	index_labels = list(range(1,len(Names)+1))
	
	# create classification structure
	makeClassificationStructure(tract_indices,sft.streamlines,clusterDict,Names,index_labels) 

if __name__ == '__main__':
	main()
