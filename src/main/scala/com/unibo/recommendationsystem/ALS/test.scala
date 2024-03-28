package com.unibo.recommendationsystem.ALS

import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import java.io.{File, PrintWriter}

object test {
  def main(args: Array[String]): Unit = {
    //recommend type, use --M for recommend users from a game or --U for recommend games from a user
    val recommendType = "--U"
    val inputID = "607050" // input id or game id
    //607050 user
    println("---------Initializing Spark-----------")
    val sparkConf = new SparkConf()
      .setAppName("Recommend")
      .set("spark.ui.showConsoleProgress", "false")
      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    println("master=" + sc.master)

    println("----------Setting Up Data Path----------")
    val dataPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv" // local for test
    val modelPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/out" // local for test


    //val checkpointPath = "home/caihao/Downloads/ml-100k/checkpoint/" // local for test
    //val dataPath = "hdfs://localhost:9000/user/caihao/movie/u.item" // HDFS
    //val modelPath = "hdfs://localhost:9000/user/caihao/movie/ALSmodel" // HDFS
    //val checkpointPath = "hdfs://localhost:9000/user/caihao/checkpoint/" // HDFS
    //sc.setCheckpointDir(checkpointPath) // checkpoint directory (to avoid stackoverflow error)

    def prepareData(sc: SparkContext, dataPath: String): RDD[(Int, String)] = {
      println("Loading Data......")
      // reads data from dataPath into Spark RDD.
      val itemRDD: RDD[String] = sc.textFile(dataPath)
      // only takes in first two fields (movieID, movieName).
      val movieTitle: RDD[(Int, String)] = itemRDD.filter(line => !line.startsWith("app_id"))
        .map(line => line.split(",")).map(x => (x(0).toInt, x(1)))
      // return movieID->movieName map as Spark RDD
      movieTitle
    }

    def loadModel(sc: SparkContext, modelPath: String): Option[MatrixFactorizationModel] = {
      try {
        val model: MatrixFactorizationModel = MatrixFactorizationModel.load(sc, modelPath)
        Some(model)
      }
      catch {
        case _: Exception => None
      }
      finally {}
    }

    def recommend(model: Option[MatrixFactorizationModel], movieTitle: RDD[(Int, String)], arg1: String, arg2: String): Unit = {
      if (arg1 == "--U") {
        recommendMovies(model.get, movieTitle, arg2.toInt)
      }
      if (arg1 == "--M") {
        recommendUsers(model.get, movieTitle, arg2.toInt)
      }
    }

    def recommendMovies(model: MatrixFactorizationModel, movieTitle: RDD[(Int, String)], inputUserID: Int): Unit = {
      val recommendP = model.recommendProducts(inputUserID, 10)
      val out = new StringBuilder()
      out.append(s"Recommending the following games for user \n ${inputUserID.toString}:")
      recommendP.foreach(p => {
        val gameTitle = movieTitle.lookup(p.product).mkString
        out.append(s"User: ${p.user}, recommended game: $gameTitle, rating: ${p.rating}\n")
      })

      out.append(s"\n")
      saveRecommendations(out, "user", inputUserID.toString)
    }

    def recommendUsers(model: MatrixFactorizationModel, movieTitle: RDD[(Int, String)], inputGameID: Int): Unit = {
      val recommendU = model.recommendUsers(inputGameID, 10)
      val out = new StringBuilder()
      out.append(s"Recommending the following users for game ${inputGameID.toString}:")
      println(s"Recommending the following users for game ${inputGameID.toString}:")
      recommendU.foreach(u => println(s"Game: ${movieTitle.lookup(u.product).mkString}, recommended user: ${u.user}, rating: ${u.rating}"))
      recommendU.foreach(u => {
        val gameTitle = movieTitle.lookup(u.product).mkString
        out.append(s"Game: $gameTitle, recommended user: ${u.user}, rating: ${u.rating}\n")
      })

      out.append(s"\n")
      saveRecommendations(out, "game", inputGameID.toString)
    }

    def saveRecommendations(out: StringBuilder, out_type: String, id: String): Unit = {
      // Crea il file
      val dir = modelPath
      val outputFile = new File(dir, "results.txt")
      val writer = new PrintWriter(outputFile)

      // Scrivi il contenuto di "out" nel file
      try {
        writer.write(out.toString())
      } finally {
        writer.close()
      }

      println("---------Preparing Data---------")
      val movieTitle: RDD[(Int, String)] = prepareData(sc, dataPath)
      //movieTitle.checkpoint() // checkpoint data to avoid stackoverflow error

      println("---------Loading Model---------")
      val model = loadModel(sc, modelPath)

      println("---------Recommend---------")
      recommend(model, movieTitle, recommendType, inputID)

      sc.stop()
    }
  }
}
