package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class mllibRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param targetUser Int, The user ID for whom the recommendations are being generated
   *
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (aggregateData, userGamesData) = timeUtils.time(preprocessData(), "Preprocessing Data", "MlLib")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(aggregateData), "Calculating TF-IDF", "MlLib")

    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "MlLib")
    topUsersSimilarity.printSchema()
    topUsersSimilarity.take(10).foreach(println)
    spark.stop()
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(userGamesData, topUsersSimilarity, targetUser), "Generating Recommendations", "MlLib")
  }

  /**
   * Preprocesses the input data to create intermediate dataframes needed for further calculations.
   *
   * @return A tuple of:
   *         - [Int, WrappedArray(String)] that maps each user with their tags for TF-IDF calculation
   *         - [Int, Int, String, String, ...] that contains game/user associations, game title and its relative tags
   *
   */
  private def preprocessData(): (DataFrame, DataFrame) = {
    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    val merged = selectedRec
      .join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    val cleanMerge = merged
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags")))
      .drop("tags")
      .cache()

    val tokenizedData = cleanMerge.withColumn("words", split(col("tagsString"), ","))

    val aggregateData = tokenizedData
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))
      .cache()

    (aggregateData, cleanMerge)
  }

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param aggregateData that contains each user and their tags with associated TF-IDF values
   * @return DataFrame (Int, Array[String], (Int, Array[Int], Array[Double]), (Int, Array[Int], Array[Double])) with these elements:
   *          - Int User ID
   *          - Array[Int] of original tags
   *          - (Int, Array[Int], Array[Double]) which contains
   *              - Int Features number
   *              - Array[Int] Tags indexes after hashing
   *              - Array[Double] Tags scores TF
   *          - (Int, Array[Int], Array[Double]) which contains
   *              - Int Features number
   *              - Array[Int] Tags indexes after hashing
   *              - Array[Double] Tags scores TF-IDF
   *
   */
  private def calculateTFIDF(aggregateData: DataFrame): DataFrame = {
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("hashedFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(aggregateData).cache()

    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    idfModel.transform(featurizedData)
  }

  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param rescaledData Map[Int, Map[String, Double] ], tf-idf score map for each userId
   * @param targetUser Int, the ID of the target user
   * @return A list of the top 3 most similar user IDs
   *
   */
  private def computeCosineSimilarity(rescaledData: DataFrame, targetUser: Int): DataFrame = {
    import spark.implicits._

    val targetUserFeatures = rescaledData
      .filter($"user_id" === targetUser)
      .select("features")
      .first()
      .getAs[Vector]("features")

    val cosineSimilarity = udf { (otherVector: Vector) =>
      val dotProduct = targetUserFeatures.dot(otherVector)
      val normA = Vectors.norm(targetUserFeatures, 2)
      val normB = Vectors.norm(otherVector, 2)
      dotProduct / (normA * normB)
    }

    rescaledData
      .filter($"user_id" =!= targetUser)
      .withColumn("cosine_sim", cosineSimilarity(col("features")))
      .select("user_id", "cosine_sim")
      .orderBy($"cosine_sim".desc)
      .limit(3)

 }

  /**
   * Generates and prints final game recommendations for a target user based on games played by similar users
   *
   * @param userGamesData [Int, Int, String, String, ...] that contains game/user associations, game title and its relative tags
   * @param usersSimilarity List[Int], list of IDs of the most similar users
   * @param targetUser Int, the ID of the target user
   *
   */
  private def generateFinalRecommendations(userGamesData: DataFrame, usersSimilarity: DataFrame, targetUser: Int): Unit = {
    import spark.implicits._

    val titlesPlayedByTargetUser = userGamesData
      .filter($"user_id" === targetUser)
      .select("title")
      .distinct()
      .as[String]
      .collect()

    val userIdsToFind = usersSimilarity
      .select("user_id")
      .as[Int]
      .collect()
      .toSet

    val finalRecommendations = userGamesData
      .filter(col("user_id").isin(userIdsToFind.toSeq: _*) && !col("title").isin(titlesPlayedByTargetUser: _*))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    finalRecommendations.show(finalRecommendations.count().toInt ,truncate = false)
  }
}