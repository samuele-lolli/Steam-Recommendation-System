package com.unibo.recommendationsystem

import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
object test {
  def main(args: Array[String]): Unit = {
    //recommend type, use --M for recommend users from a game or --U for recommend games from a user
    val recommendType = "--U"
    val inputID = "51580" // input id or game id

    println("---------Initializing Spark-----------")
    val sparkConf = new SparkConf()
      .setAppName("Recommend")
      .set("spark.ui.showConsoleProgress", "false")
      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    println("master=" + sc.master)

    println("----------Setting Up Data Path----------")
    val dataPath = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\games.csv" // local for test
    val modelPath = "C:\\Users\\samue\\recommendationsystem\\out" // local for test
    //val checkpointPath = "home/caihao/Downloads/ml-100k/checkpoint/" // local for test
    //val dataPath = "hdfs://localhost:9000/user/caihao/movie/u.item" // HDFS
    //val modelPath = "hdfs://localhost:9000/user/caihao/movie/ALSmodel" // HDFS
    //val checkpointPath = "hdfs://localhost:9000/user/caihao/checkpoint/" // HDFS
    //sc.setCheckpointDir(checkpointPath) // checkpoint directory (to avoid stackoverflow error)

    def prepareData(sc: SparkContext, dataPath:String): RDD[(Int, String)] ={
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

    def recommend(model: Option[MatrixFactorizationModel], movieTitle:RDD[(Int, String)], arg1: String, arg2: String): Unit ={
      if (arg1 == "--U") {
        recommendMovies(model.get, movieTitle, arg2.toInt)
      }
      if (arg1 == "--M") {
        recommendUsers(model.get, movieTitle, arg2.toInt)
      }
    }

    def recommendMovies(model: MatrixFactorizationModel, movieTitle: RDD[(Int, String)], inputUserID: Int): Unit = {
      val recommendP = model.recommendProducts(inputUserID, 10)
      println(s"Recommending the following games for user ${inputUserID.toString}:")
      recommendP.foreach(p => println(s"user: ${p.user}, recommended movie: ${movieTitle.lookup(p.product).mkString}, rating: ${p.rating}"))
    }

    def recommendUsers(model: MatrixFactorizationModel, movieTitle: RDD[(Int, String)], inputMovieID: Int): Unit = {
      val recommendU = model.recommendUsers(inputMovieID, 10)
      println(s"Recommending the following users for game ${inputMovieID.toString}:")
      recommendU.foreach(u => println(s"movie: ${movieTitle.lookup(u.product).mkString}, recommended user: ${u.user}, rating: ${u.rating}"))
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