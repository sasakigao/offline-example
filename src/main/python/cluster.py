
"""
Decision Tree Classification Example.
"""

import time
from math import sqrt
import sys
sys.path.append('/home/sasaki/devapp/spark-1.6.1-bin-hadoop2.6/python/lib')
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.clustering import KMeansModel, KMeans
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

appName="Binary Classifier"
# Read from
rawDataHDFS = "file:///home/sasaki/dev/gamease/dimensions-verify/res/20170118-0955/"
# Write to
clusterDataHDFS = "file:///home/sasaki/dev/gamease/example/res/cluster/data/"
clusterModelHDFS = "file:///home/sasaki/dev/gamease/example/res/cluster/model/"


def line2Feature(line):
	values = [float(x) for x in line]
	return values

def error(point, clusters):
	    center = clusters.centers[clusters.predict(point)]
	    return sqrt(sum([x**2 for x in (point - center)]))

if __name__ == "__main__":
    	timeformat = "%Y%m%d-%H%M"
    	timeStr = time.strftime(timeformat, time.localtime())

	confSpark = SparkConf().setAppName(appName)
    	sc = SparkContext(conf = confSpark)

	rawData = sc.textFile(rawDataHDFS).map(lambda x : x.split(","))
   	features = rawData.map(lambda x : line2Feature(x[2:]))
	clusters = KMeans.train(features, 2, maxIterations=20, runs=10, 
		initializationMode='k-means||', seed=21, initializationSteps=5, 
		epsilon=0.0001, initialModel=None)	

	WSSSE = features.map(lambda point: error(point, clusters)).reduce(lambda x, y : x + y)
	# clusters.save(clusterModelHDFS)
	# clusters = Loader.load(clusterModelHDFS)
    	
    	# (prediction, label, uid)
    	predictions = rawData.map(lambda x : (clusters.predict(line2Feature(x[2:])), x[1], x[0]))
    	predictions.coalesce(5, False).saveAsTextFile(clusterDataHDFS + timeStr)
	sc.stop()