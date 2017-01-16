import scala.collection.mutable._

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, Row}
import org.apache.spark.sql.types._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.tree.model.{DecisionTreeModel}

import java.util.Date
import java.text.SimpleDateFormat


object Trainer {
	val appName = "Binary Classifier"

    	// Read from
	val featuresHDFS = "hdfs:///netease/feature/"
	val treeModelPathHDFS = "hdfs:///netease/model/tree/"
	val forestModelPathHDFS = "hdfs:///netease/model/forest/"
	val gbtModelPathHDFS = "hdfs:///netease/model/gbt/"
	
	// Write to
	val treeMetricsPathHDFS = "hdfs:///netease/ver2/dim-verify/metrics/tree/"

	def main(args: Array[String]) = {
	  	val confSpark = new SparkConf().setAppName(appName)
	  	confSpark.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  		val sc = new SparkContext(confSpark)

  		val rawFeatures = MLUtils.loadLibSVMFile(sc, featuresHDFS, 100)

		val timeStr = getTime
  		val modelTree = DecisionTreeModel.load(sc, treeModelPathHDFS + timeStr)
  		val predicationsAndLabels = rawFeatures.map{ point =>
 	 		val predication = modelTree.predict(point.features)
  			(predication, point.label)
		}
		binaryPredicationMetricsOutput(predicationsAndLabels, treeMetricsPathHDFS + timeStr, sc)
		sc.stop()
  	}

  	def binaryPredicationMetricsOutput(predicationsAndLabels : RDD[(Double, Double)], 
  			metricsPath : String, sc : SparkContext) = {
  		predicationsAndLabels.persist()
  		// some measurements for binary classifier
  		val accuracy = 1.0 * predicationsAndLabels.filter(x => x._1 == x._2).count / predicationsAndLabels.count
  		val precision = 1.0 * predicationsAndLabels.filter(x =>x._1 == x._2 && x._1 == 1.0).count / 
  			predicationsAndLabels.filter(_._1 == 1.0).count
  		val recall = 1.0 * predicationsAndLabels.filter(x =>x._1 == x._2 && x._1 == 1.0).count / 
  			predicationsAndLabels.filter(_._2 == 1.0).count
  		val f1Measure = 2 * precision * recall / (precision + recall)
  		val metrics = List(("accuracy", accuracy), ("precision", precision), ("recall", recall), ("f1Measure", f1Measure))
		sc.parallelize(metrics, 1).saveAsTextFile(metricsPath)
		predicationsAndLabels.unpersist(true)
  	}

  	def getTime = {
  		val dt = new Date()
     		val sdf = new SimpleDateFormat("yyyyMMdd-HHmm")
     		sdf.format(dt)
  	}
}
