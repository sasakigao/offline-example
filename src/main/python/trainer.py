
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


appName="Binary Classifier"
# Read from
featuresHDFS = "file:///home/sasaki/dev/gamease/example/res/offline/features/20170110-1645"
treeModelPathHDFS = "file:///home/sasaki/dev/gamease/example/res/offline/tree/20170110-1645"
# Write to
treeMetricsPathHDFS = "file:///home/sasaki/metrics/"

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


if __name__ == "__main__":
    	timeformat = "%Y%m%d-%H%M"
    	timeStr = time.strftime(timeformat, time.localtime())

	confSpark = SparkConf().setAppName(appName)
    	sc = SparkContext(conf=confSpark)
    	features = MLUtils.loadLibSVMFile(sc, featuresHDFS)
    	modelTree = DecisionTreeModel.load(sc, treeModelPathHDFS)

    	# In python the mllib model seems unable to be directly used in a closure.
    	predictions = modelTree.predict(features.map(lambda x: x.features))
    	predicationsAndLabels = predictions.zip(features.map(lambda x: x.label)).persist()
	binary_redication_metrics_output(predicationsAndLabels, treeMetricsPathHDFS + timeStr, sc)
	predicationsAndLabels.persist()

	sc.stop()