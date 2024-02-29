package com.unibo.recommendationsystem

import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
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
      println(ratingsRDD.first()) // Rating(196,242,3.0)
      // return processed data as Spark RDD
      ratingsRDD
    }



    def saveModel(context: SparkContext, model:MatrixFactorizationModel, modelPath: String): Unit ={
      try {
        model.save(context, modelPath)
      }
      catch {
        case e: Exception => println("Error Happened when saving model!!!" + e )
      }
      finally {
      }
    }


    val hashingTF = new HashingTF()
    val tf = hashingTF.transform(data)



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
      val err = (r1 - r2)
      err * err
    }.mean()
    println(s"Mean Squared Error = $MSE")


    println("----------Saving Model---------")
    saveModel(sc, model, modelPath)
    sc.stop()
  }
}

    /*
    /*Caricamento dei dati nella forma originale */
    val rawData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(csvFilePath)


     */
/*

   // val limitedData = rawData.limit(1000000)
   val columnsToKeep = Seq("user_id", "app_id", "hours")
    val newDF = rawData.select(columnsToKeep.map(col): _*)
    newDF.write.csv("/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/output.csv")



    val csvFilePath2 = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/output.csv"

    val ratingsRows = spark.sparkContext.textFile(csvFilePath2).map(line => line.split(",") match {
      case Array(user, item, rate) => Row(user.toInt, item.toInt, rate.toDouble)
    })
/*
    val schema = StructType(Seq(
      StructField("userId", IntegerType, false),
      StructField("itemId", IntegerType, false),
      StructField("rating", DoubleType, false)
    ))

 */

    // Costruzione del DataFrame con i dati necessari (user_id, app_id, hours_played)
  /*  val ratings = limitedData.select("user_id", "app_id", "hours")
      .withColumnRenamed("user_id", "userId")
      .withColumnRenamed("app_id", "itemId")
      .withColumnRenamed("hours", "rating")
      .na.drop() // Rimuove eventuali righe con valori nulli

   */
   // val Array(training, test) = ratingsDF.randomSplit(Array(0.8, 0.2))

    val model : MatrixFactorizationModel = ALS.train()

    // Creazione del modello ALS (Alternating Least Squares)
    val als = new ALS()
      .setMaxIter(20)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("userId")
      .setRatingCol("rating")

    val model = als.fit(trainingDF)

    // Valutazione del modello
    model.setColdStartStrategy("drop")

    val predictions = model.transform(testDf)
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    import spark.implicits._
    // Ottenere i vicini dell'utente dato
    val user = training.select("userId").distinct().limit(1)

    val userId = user.collect().head.getAs[Int]("userId")

    /*val itemsPlayed = training.filter($"userId" === userId)
      .select("itemId")

    println("itemsPlayed")
    println(itemsPlayed.show())

     */


    val nNeighbors = 5
    val userRecs = model.recommendForUserSubset(user, nNeighbors)

    userRecs.show() // 'items' are actually similar users

    //stampo tutti gli utenti raccomandati
    println("Raccomandazione:")
    userRecs.collect.foreach(println)

    // Stampare i vicini dell'utente

    spark.stop()

  }
}
*/

//provare a stampare più raccomandazioni

//MSE con 3 milioni di righe di codice 95, .....





//Collaborative filtering with  Kmeans
/*
object main {
  def main(args: Array[String]): Unit = {

    /* Inizializzazione SparkSession */
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512m")
      .getOrCreate()

    /* Path per il dataset */
    val csvFilePath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"

    /* Caricamento dei dati nella forma originale */
    val rawDataSplitted = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(csvFilePath)



    /*   val splitWeights = Array.fill(10)(0.01)
    val splits = rawData.randomSplit(splitWeights)
    val rawDataSplit = splits(0)

 */

    /* val singlePartitionDF = rawDataSplit.repartition(1)
    singlePartitionDF.write
      .option("header", "true")
      .csv("/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/piccolo")

    */

    /*val rawDataSplitted = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/")

    */

*/

    //val columnLength = limitedData.select(functions.count("app_id")).collect()(0)
   /* spark.conf.set("spark.sql.pivotMaxValues", 1000000)
   // println("Column length: " + columnLength)


//291550
    //Pivoting
    //val pivotedData = limitedData.groupBy("user_id")
    val pivotedData = ratings.groupBy("userId")
      .pivot("itemId")
      .sum("rating")

//Println per controllo valori null
    println("Pivoted data: " + pivotedData.show())

    // Missing Value Handling, Fill with Zeros
    val filledData = pivotedData.na.fill(0)
    println("Filled data: " + filledData.show())

    val assembler = new VectorAssembler()
      .setInputCols(filledData.columns.filter(_ != "userId"))
      .setOutputCol("features")
    val featureData = assembler.transform(filledData)

    println("featured data: " + featureData.show())

    val k_means = new KMeans().setK(10) // Adjust 'k' (number of clusters) as needed

    println("Fit the model")

    val model = k_means.fit(featureData)

    println("Finish to fit the model")

    //test visualizzazione utente
    val targetUserId = featureData.head.getAs[Int]("userId")
    println("Target user getAs[Int]: " + targetUserId)


    val indexer = new StringIndexer().setInputCol("userId").setOutputCol("user_id_index")
    val indexedModel = indexer.fit(featureData)
    val labels = indexedModel.labelsArray // Extract the array of original labels
    println("Labels" + labels.mkString(", "))

   /* def getOriginalUserId(index: Int, labelIndex: Int = 0): String = {
      println(s"Index: $index, labelIndex: $labelIndex")
      if (index < labels.length && labelIndex < labels(index).length) {
        labels(index)(labelIndex)
      } else {
        "Label Not Found"  // Handle out-of-bounds cases
      }
    }
    */

    val finalModel = indexedModel.transform(featureData)
    val predictions = model.transform(finalModel)

    println("Pred: " + predictions.show())


    // Get feature vector for the target user
    val targetUserRow = predictions.where(col("userId") === targetUserId).select("features").head()
    val targetUserVector = targetUserRow.getAs[org.apache.spark.ml.linalg.Vector](0)

    println("Prediction start")
    val targetClusterId = model.predict(targetUserVector)

    val recommendations = predictions.filter(col("prediction") === targetClusterId)
      .select("userId")
      .where(col("userId") =!= targetUserId)
      .limit(5)
      .collect()
      .map(_.getAs[Int](0))
    //.map(row => getOriginalUserId(row.getAs[Int]("user_id")))

    println(s"Recommendations for $targetUserId: ${recommendations.mkString(", ")}")

    for (recommendation <- recommendations) {
      println(s"Recommendation: $recommendation")
    }

    println("Recommendations:")
    recommendations.foreach(println)*/





    /* Pre-processing dei dati (a cosa serve sta roba? Mistero..) */
   /* val userIndexer = new StringIndexer()
      .setInputCol("user_id")
      .setOutputCol("user_id_indexed")
    val userIndexedData = userIndexer.fit(rawDataSplitted).transform(rawDataSplitted)

    val appIndexer = new StringIndexer()
      .setInputCol("app_id")
      .setOutputCol("app_id_indexed")
    val data = appIndexer.fit(userIndexedData).transform(userIndexedData)


    import spark.implicits._
    val interactions = data.select("user_id_indexed", "app_id_indexed", "hours")
      .rdd.map(r => (r.getInt(0), r.getInt(1), r.getDouble(2)))

    val maxAppId = interactions.map(_._2).reduce(Math.max)

    val interactionsWithSparse = interactions.map { case (userId, appId, rating) =>
        (userId, Vectors.sparse(maxAppId, Seq((appId, rating))))
      }
      .toDF("user_id_indexed", "features")



    /* Inizializzazione e training del modello ALS (KNN) */
    println("INIZIO DEL TRAINING DI ALS MODEL.........")
    val als = new ALS()
      .setUserCol("user_id_indexed")
      .setItemCol("features")
      //.setRatingCol("hours")
      .setRank(10)
      .setMaxIter(10)
      .setRegParam(0.1)
      .setImplicitPrefs(true)

    val model = als.fit(interactionsWithSparse)

    //model.setColdStartStrategy("drop")


    /* Generazione di 5 raccomandazioni per l'utente con id=0 */
    import spark.implicits._
    val userId = 0
    val userSubset = Seq(userId).toDF("user_id_indexed") // Create the DataFrame
    val recommendations = model.recommendForUserSubset(userSubset, 5)
    recommendations.show()
    spark.stop()

    */

