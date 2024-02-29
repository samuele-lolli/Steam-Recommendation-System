package com.unibo.recommendationsystem

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.dsl.expressions.intToLiteral

object mainTf {

  def main(args: Array[String]): Unit = {

    println("---------Initializing Spark-----------")
    val sparkConf = new SparkConf()
      .setAppName("RecommenderTrain")
      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    println("master=" + sc.master)

    println("----------Setting up data path------------")
    val dataPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv" // local for test
    val modelPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/out" // local for test
    val checkpointPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/src/main/scala/com/unibo/recommendationsystem" // local for test
    //sc.setCheckpointDir(checkpointPath) // checkpoint directory (to avoid stackoverflow error)



    def preprocessing(sc: SparkContext, dataPath: String):  RDD[(Int, Map[Int, Double])] = {
      // reads data from dataPath into Spark RDD-
      val file: RDD[String] = sc.textFile(dataPath)

      //Create an RDD[(Int,Int,Double)]
      val splitRDD = file.map(line => line.split(","))
      val header = file.first() // Get the first line (header)
      val columnIndices = header.split(",").zipWithIndex.toMap
      val dataRDD = splitRDD.filter(lineArray => lineArray.mkString(",") != header) // Filter out the header row
        .map(row => (row(columnIndices("user_id")).toInt,
          row(columnIndices("app_id")).toInt,
          row(columnIndices("hours")).toDouble))

      val firstRow = dataRDD.first()
      println("---------Print first row from  RDD[(Int, Int, Double)---------")
      println(s"(${firstRow._1}, ${firstRow._2}, ${firstRow._3})")

      //(49625, 975370, 36.3)

      //Sum multiple (userId, appId, hours) entries for the same user-app pair
      val aggregatedRDD = dataRDD.map { case (userId, appId, hours) => ((userId, appId), hours) }
        .reduceByKey(_ + _) // Aggregate playtime by (userId, appId)
        .map { case ((userId, appId), totalHours) => (userId, (appId, totalHours)) }
      val groupedRDD = aggregatedRDD.groupByKey() // RDD[(Int, Iterable[(Int, Double)])]

      //each element is a (userId, playtimeDictionary).
      val playtimeDictRDD = groupedRDD.mapValues { iterable =>
        iterable.toMap
      } // RDD[(Int, Map[Int, Double])]

      println("---------Print first row from RDD[(Int, Map[Int, Double])]---------")
      val firstRow2 = playtimeDictRDD.first() // Assuming your RDD is named 'dataRDD'
      // Print the userId (key)
      println(s"userId: ${firstRow2._1}")
      // Print the appId, playtime pairs (value)
      println("appId, playtime pairs:")
      firstRow2._2.foreach { case ( appId, playtime) =>
        println(s"   appId: $appId, hours: $playtime")
      }
      /*
      userId: 12228438
    appId, playtime pairs:
    appId: 414700, hours: 8.8
    appId: 292030, hours: 65.3
    appId: 883710, hours: 19.2
       */

      playtimeDictRDD
    }

    println("---------Preparing Data---------")
    val ratingsRDD:  RDD[(Int, Map[Int, Double])] = preprocessing(sc, dataPath)
    //ratingsRDD.checkpoint() // checkpoint data to avoid stackoverflow error





  }
}
