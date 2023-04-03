import numpy as np
from sklearn.utils import shuffle

# Generates 500 objects of the following format,
# {
#  "ASV": X, -> Ranges between [1, 4] uniformly distributed.
#  "downstreamASV": Y, -> Ranges between [1, 4] uniformly distributed.
#  "responseTime": Z, -> N(1500, 150)
#  "responseCode": W, -> In [200, 301, 302, 404, 410, 500, 503]
#  "downstreamResponseTime": T, -> N(1500, 150)
#  "latencyMilliseconds": U -> N(9500, 200)
# }

def generateData(RESPONSE_TIME_MEAN=1500, RESPONSE_TIME_VAR=150, \
		 DOWNSTREAM_RESPONSE_TIME_MEAN=1500, DOWNSTREAM_RESPONSE_TIME_VAR=150, \
			LATENCY_MILLISECONDS_MEAN=9500, LATENCY_MILLISECONDS_VAR=200,\
				amount=500): 
	"""
	Generates 500 rows of the following format,
	ASV, downstreamASV, responseTime, responseCode, downstreamResponseTime, latencyMilliseconds

	Can alter the mean and variance of the responseTime, downstreamResponseTime, and 
	latencyMilliseconds by passing in the appropriate parameters.
	"""
	data = []
	for i in range(amount):
		data.append({
			"ASV": np.random.randint(1, 5),
			"downstreamASV": np.random.randint(1, 5),
			"responseTime": np.random.normal(RESPONSE_TIME_MEAN, RESPONSE_TIME_VAR),
			"responseCode": np.random.choice([200, 301, 302, 404, 410, 500, 503]),
			"downstreamResponseTime": np.random.normal(DOWNSTREAM_RESPONSE_TIME_MEAN, DOWNSTREAM_RESPONSE_TIME_VAR),
			"latencyMilliseconds": np.random.normal(LATENCY_MILLISECONDS_MEAN, LATENCY_MILLISECONDS_VAR)
		})
	return data

# Make Data
def make_data():
	# Will generate data for 3 classes, 
	# Class 1: higher responseTime with lower latency than default.
	# Class 2: higher latency.
	# Class 3: Low downstreamResponseTime with High responseTime.

	# Class 1
	class1 = generateData(RESPONSE_TIME_MEAN=3000, RESPONSE_TIME_VAR=100, \
			LATENCY_MILLISECONDS_MEAN=5000, LATENCY_MILLISECONDS_VAR=100)
	
	class1_label = [1] * len(class1)

	# Class 2
	class2 = generateData(LATENCY_MILLISECONDS_MEAN=15000, LATENCY_MILLISECONDS_VAR=200)

	class2_label = [2] * len(class2)

	# Class 3
	class3 = generateData(RESPONSE_TIME_MEAN=3000, RESPONSE_TIME_VAR=100, \
			DOWNSTREAM_RESPONSE_TIME_MEAN=500, DOWNSTREAM_RESPONSE_TIME_VAR=100)
	
	class3_label = [3] * len(class3)

	# Combine all classes
	data = class1 + class2 + class3
	labels = class1_label + class2_label + class3_label

	# Shuffle the data
	data, labels = shuffle(data, labels, random_state=0)

	return data, labels



# Write to csv 
def writeData(data):
	with open('data.csv', 'w') as f:
		f.write("ASV,downstreamASV,responseTime,responseCode,downstreamResponseTime,latencyMilliseconds\n")
		for d in data:
			f.write("{},{},{},{},{},{}\n".format(d["ASV"], d["downstreamASV"], d["responseTime"], d["responseCode"], d["downstreamResponseTime"], d["latencyMilliseconds"]))
		f.close()

def writeLabel(labels):
	with open('labels.csv', 'w') as f:
		f.write("label\n")
		for l in labels:
			f.write("{}\n".format(l))
		f.close()

data, labs = make_data()
writeData(data)
writeLabel(labs)

