import numpy as np


# Generates 500 objects of the following format,
# {
#  "ASV": X, -> Ranges between [1, 4] uniformly distributed.
#  "downstreamASV": Y, -> Ranges between [1, 4] uniformly distributed.
#  "responseTime": Z, -> N(1500, 150)
#  "responseCode": W, -> In [200, 301, 302, 404, 410, 500, 503]
#  "downstreamResponseTime": T, -> N(1500, 150)
#  "latencyMilliseconds": U -> N(9500, 200)
# }

def generateData():
	data = []
	for i in range(500):
		data.append({
			"ASV": np.random.randint(1, 5),
			"downstreamASV": np.random.randint(1, 5),
			"responseTime": np.random.normal(1500, 150),
			"responseCode": np.random.choice([200, 301, 302, 404, 410, 500, 503]),
			"downstreamResponseTime": np.random.normal(1500, 150),
			"latencyMilliseconds": np.random.normal(9500, 200)
		})
	return data


# Write to csv 
def writeData(data):
	with open('data.csv', 'w') as f:
		f.write("ASV,downstreamASV,responseTime,responseCode,downstreamResponseTime,latencyMilliseconds\n")
		for d in data:
			f.write("{},{},{},{},{},{}\n".format(d["ASV"], d["downstreamASV"], d["responseTime"], d["responseCode"], d["downstreamResponseTime"], d["latencyMilliseconds"]))
		f.close()

writeData(generateData())
