package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class mllibRecommendation(spark: SparkSession, dfRec: Dataset[Row], dfGames: DataFrame, dfMetadata: DataFrame) {

  /**
   * Generates game recommendations for a target user.
   *
   * @param targetUser The ID of the user for whom recommendations are to be generated.
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (preprocessedData, gameTitles, userGameDetails) = timeUtils.time(preprocessData(), "Preprocessing Data", "MLLIB")

    println("Calculate term frequency and inverse document frequency...")
    val tfidf = timeUtils.time(calculateTFIDF(preprocessedData), "Calculating TF-IDF", "MLLIB")

    println("Calculate cosine similarity to get similar users...")
    val similarUsers = timeUtils.time(computeCosineSimilarity(tfidf, targetUser), "Getting Similar Users", "MLLIB")

    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(similarUsers, targetUser, gameTitles, userGameDetails), "Generating Recommendations", "MLLIB")
  }

  /**
   * Preprocesses the input data by normalizing tags and preparing data for TF-IDF.
   *
   * @return A tuple containing:
   *         - A DataFrame with user and their associated tag words.
   *         - A DataFrame mapping game IDs to their titles.
   *         - A DataFrame with detailed information about games and users.
   */
  private def preprocessData(): (DataFrame, DataFrame, DataFrame) = {
    val userGameDetails = dfRec.select("app_id", "user_id")
      .join(dfGames.select("app_id", "title"), Seq("app_id"))
      .join(dfMetadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)
      .withColumn("normalized_tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tags_string", concat_ws(",", col("normalized_tags")))
      .drop("tags")
      .drop("normalized_tags")

    val userTagData = userGameDetails
      .withColumn("tag_words", split(col("tags_string"), ","))
      .groupBy("user_id")
      .agg(flatten(collect_list("tag_words")).as("tag_words"))
      .persist(StorageLevel.MEMORY_AND_DISK)

    val gameTitles = dfGames.select("app_id", "title")

    (userTagData, gameTitles, userGameDetails)
  }

  /**
   * Calculates the Term Frequency-Inverse Document Frequency (TF-IDF) for tags associated with users.
   *
   * @param userTagData A DataFrame containing users and their associated tags.
   * @return A DataFrame containing TF-IDF features for each user.
   */
  private def calculateTFIDF(userTagData: DataFrame): DataFrame = {
    val hashingTF = new HashingTF().setInputCol("tag_words").setOutputCol("hashedFeatures").setNumFeatures(20000)
    val tf = hashingTF.transform(userTagData)

    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(tf)
    idfModel.transform(tf)
  }

  /**
   * Computes the cosine similarity between a target user and other users based on TF-IDF vectors.
   *
   * @param tfidf A DataFrame containing TF-IDF features for all users.
   * @param targetUser   The ID of the target user.
   * @return A list of user IDs similar to the target user, ordered by similarity.
   */
  private def computeCosineSimilarity(tfidf: DataFrame, targetUser: Int): List[Int] = {
    import spark.implicits._

    // Fetch the target user's feature vector
    val targetUserFeatures = tfidf
      .filter($"user_id" === targetUser)
      .select("features")
      .first()
      .getAs[Vector]("features")

    // Precompute the norm of the target user's feature vector
    val targetUserNorm = Vectors.norm(targetUserFeatures, 2)

    val targetUserFeaturesBroadcast = spark.sparkContext.broadcast(targetUserFeatures)

    //Compute cosine similarity
    val topSimilarUsers = tfidf
      .filter($"user_id" =!= targetUser)
      .map { row =>
        val userId = row.getAs[Int]("user_id")
        val features = row.getAs[Vector]("features")
        val dotProduct = targetUserFeaturesBroadcast.value.dot(features)
        val normB = Vectors.norm(features, 2)
        val cosineSim = if (normB != 0) dotProduct / (targetUserNorm * normB) else 0.0
        (userId, cosineSim)
      }

      // Convert to DataFrame, sort, and pick top users
      .toDF("user_id", "cosine_sim")
      .orderBy($"cosine_sim".desc)
      .limit(3)
      .select("user_id")
      .as[Int]
      .collect()

    // Release the broadcast variable
    targetUserFeaturesBroadcast.unpersist()

    topSimilarUsers.toList
  }


  /**
   * Generates the final recommendations for the target user
   *
   * @param similarUserIds  A List[Int] containing the top 3 similar users ids
   * @param targetUser The ID of the target user.
   * @param gameTitles A DataFrame mapping game IDs to their titles.
   * @param userGameDetails - A DataFrame with detailed information about games and users.
   */
  private def generateFinalRecommendations(similarUserIds: List[Int], targetUser: Int, gameTitles: DataFrame, userGameDetails: DataFrame): Unit = {
    val gamesPlayedBySimilarUsers = userGameDetails
      .filter(col("user_id").isin(similarUserIds: _*))
      .select("app_id", "user_id")

    val gamesPlayedByTargetUser = userGameDetails
      .filter(col("user_id") === targetUser)
      .select("app_id")

    val recommendations = gamesPlayedBySimilarUsers
      .join(gamesPlayedByTargetUser, Seq("app_id"), "left_anti")
      .join(gameTitles, Seq("app_id"))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("similar_user_ids"))

    recommendations.show(recommendations.count().toInt, truncate = false)
  }
}
