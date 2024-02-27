package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
object main {
  def main(args: Array[String]): Unit = {

    println("---------Initializing Spark-----------")

    val sparkConf = new SparkConf()
      .setAppName("RecommenderTrain")
      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    println("master=" + sc.master)

    println("----------Setting up data path----------")
    val dataPath = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\recommendations.csv" // local for test
    val modelPath = "C:\\Users\\samue\\recommendationsystem\\src\\main\\scala\\com\\unibo\\recommendationsystem" // local for test
    //val checkpointPath = "C:\\Users\\samue\\recommendationsystem\\src\\main\\scala\\com\\unibo\\recommendationsystem" // local for test
    //sc.setCheckpointDir(checkpointPath) // checkpoint directory (to avoid stackoverflow error)

    def preprocessing(sc: SparkContext, dataPath:String): RDD[Rating] ={
      // reads data from dataPath into Spark RDD.
      val file: RDD[String] = sc.textFile(dataPath)
      // only takes in first three fields (userID, itemID, rating).
      val ratingsRDD: RDD[Rating] = file.filter(line => !line.startsWith("app_id")).map(line => line.split(",") match {
        case Array(app, _, _, _, _, hours, user, _) => Rating(user.toInt, app.toInt, hours.toDouble)
      })
      println(ratingsRDD.first()) // Rating(196,242,3.0)
      // return processed data as Spark RDD
      ratingsRDD
    }

    def saveModel(context: SparkContext, model:MatrixFactorizationModel, modelPath: String): Unit ={
      try {
        model.save(context, modelPath)
      }
      catch {
        case e: Exception => println("Error Happened when saving model!!!")
      }
      finally {
      }
    }

    println("---------Preparing Data---------")
    val ratingsRDD: RDD[Rating] = preprocessing(sc, dataPath)
    //ratingsRDD.checkpoint() // checkpoint data to avoid stackoverflow error

    println("---------Training---------")
    println("Start ALS training, rank=5, iteration=20, lambda=0.1")
    val model: MatrixFactorizationModel = ALS.train(ratingsRDD, 5, 20, 0.1)

    println("----------Saving Model---------")
    saveModel(sc, model, modelPath)
    sc.stop()
  }
}