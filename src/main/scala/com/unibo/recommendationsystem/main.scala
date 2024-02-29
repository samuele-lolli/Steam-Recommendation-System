package com.unibo.recommendationsystem

import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
object main {
  def main(args: Array[String]): Unit = {

    println("---------Initializing Spark-----------")
    val sparkConf = new SparkConf()
      .setAppName("RecommenderTrain")
      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    println("master=" + sc.master)

    println("----------Setting up data path----------")
    val dataPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv" // local for test
    val modelPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/out" // local for test
    val checkpointPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/src/main/scala/com/unibo/recommendationsystem" // local for test
    //sc.setCheckpointDir(checkpointPath) // checkpoint directory (to avoid stackoverflow error)

    def preprocessing(sc: SparkContext, dataPath:String): RDD[Rating] ={
      // reads data from dataPath into Spark RDD.
      val file: RDD[String] = sc.textFile(dataPath)
      // only takes in first three fields (userID, itemID, rating).
      val ratingsRDD: RDD[Rating] = file.filter(line => !line.startsWith("app_id")).map(line => line.split(",") match {
        case Array(app, _, _, _, _, hours, user, _) => Rating(user.toInt, app.toInt, hours.toDouble)
      })

      val limitedRatingsRDD = ratingsRDD.mapPartitions(iter => {
        val limitedRatingsArray = iter.take(1000000).toArray
        limitedRatingsArray.iterator
      })
      println(limitedRatingsRDD.first()) // Rating(196,242,3.0)
      // return processed data as Spark RDD
      limitedRatingsRDD
    }

    def saveModel(context: SparkContext, model:MatrixFactorizationModel, modelPath: String): Unit ={
      try {
        model.save(context, modelPath)
      }
      catch {
        case e: Exception => println(e)
      }
      finally {
      }
    }

    println("---------Preparing Data---------")
    val ratingsRDD: RDD[Rating] = preprocessing(sc, dataPath)
    //ratingsRDD.checkpoint() // checkpoint data to avoid stackoverflow error

    println("---------Training---------")
    println("Start ALS training, rank=5, iteration=20, lambda=0.1")
    val model: MatrixFactorizationModel = ALS.train(ratingsRDD, 25, 18, 0.1)
    //Mean Squared Error = 279, rank = 25 - iteration = 18

    //Test
    //val model: MatrixFactorizationModel = ALS.train(ratingsRDD, 10, 20, 0.1)
    //Mean Squared Error = 1033.514233485899

    // Evaluate the model on rating data
    val usersItems = ratingsRDD.map { case Rating(user, item, rate) =>
      (user, item)
    }

    val predictions =
      model.predict(usersItems).map { case Rating(user, item, rate) =>
        ((user, item), rate)
      }

    val ratesAndPreds = ratingsRDD.map { case Rating(user, item, rate) =>
      ((user, item), rate)
    }.join(predictions)

    val MSE = ratesAndPreds.map { case ((user, item), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()
    println(s"Mean Squared Error = $MSE")

    println("----------Saving Model---------")
    saveModel(sc, model, modelPath)
    sc.stop()
  }
}

