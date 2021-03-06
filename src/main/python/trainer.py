
"""
Decision Tree Classification Example.
"""

import time
import sys
sys.path.append('/home/sasaki/devapp/spark-1.6.1-bin-hadoop2.6/python/lib')
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

appName="Binary Classifier"
# Read from
rawDataHDFS = "file:///home/sasaki/Documents/qnlog/fixed/test"
featuresHDFS = "file:///home/sasaki/dev/gamease/example/res/offline/features/20170110-1645"
treeModelLoadHDFS = "file:///home/sasaki/dev/gamease/example/res/offline/tree/20170110-1645"
# Write to
featuresSaveHDFS = "file:///home/sasaki/model/features/"
treeModelSaveHDFS = "file:///home/sasaki/model/tree/"
treeMetricsPathHDFS = "file:///home/sasaki/metrics/"

def trainTree(train, sc, timeStr):
	modelTree = DecisionTree.trainClassifier(train, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
	modelTree.save(sc, treeModelSaveHDFS + timeStr)
	return modelTree

def binary_redication_metrics_output(predicationsAndLabels, metricsPath, sc):
	predicationsAndLabels.persist()
  	# some measurements for binary classifier
	accuracy = 1.0 * predicationsAndLabels.filter(lambda (x, y) : x == y).count() / \
		predicationsAndLabels.count()
	precision = 1.0 * predicationsAndLabels.filter(lambda (x, y) : (x == y) & (x == 1.0)).count() / \
		predicationsAndLabels.filter(lambda (x, y) : x == 1.0).count()
	recall = 1.0 * predicationsAndLabels.filter(lambda (x, y) : (x == y) & (x == 1.0)).count() / \
  		predicationsAndLabels.filter(lambda (x, y) : y == 1.0).count()
	f1Measure = 2 * precision * recall / (precision + recall)
	metrics = [("accuracy", accuracy), ("precision", precision), ("recall", recall), ("f1Measure", f1Measure)]
	sc.parallelize(metrics, 1).saveAsTextFile(metricsPath)
	predicationsAndLabels.unpersist()

def line2Feature(line):
	values = [float(x) for x in line.split(',')]
	return LabeledPoint(values[0], values[1:])

if __name__ == "__main__":
	# 0 means train while 1 means offline prediction
	mode = sys.argv[1]
    	timeformat = "%Y%m%d-%H%M"
    	timeStr = time.strftime(timeformat, time.localtime())

	confSpark = SparkConf().setAppName(appName)
    	sc = SparkContext(conf = confSpark)

    	if mode == "0":
	    	rawData = sc.textFile(rawDataHDFS)
	    	features = rawData.map(lambda x : line2Feature(x))
	    	splits = features.randomSplit([0.7, 0.3], 21)
	    	train = splits[0].persist()
	    	test = splits[1].persist()
	    	modelTree = trainTree(train, sc, timeStr)
	    	MLUtils.saveAsLibSVMFile(features.repartition(1), featuresSaveHDFS + timeStr)
    	else:
	    	test = MLUtils.loadLibSVMFile(sc, featuresHDFS)
	    	modelTree = DecisionTreeModel.load(sc, treeModelLoadHDFS)

    	# In python the mllib model seems unable to be directly used in a closure.
    	predictions = modelTree.predict(test.map(lambda x : x.features))
    	predicationsAndLabels = predictions.zip(test.map(lambda x : x.label)).persist()
	binary_redication_metrics_output(predicationsAndLabels, treeMetricsPathHDFS + timeStr, sc)
	predicationsAndLabels.unpersist()

	sc.stop()