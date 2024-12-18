package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class sqlRecommendation(spark: SparkSession, dfRec: Dataset[Row], dfGames: DataFrame, dfMetadata: DataFrame) {

  /**
   * Generates game recommendations for a target user.
   *
   * @param targetUser The ID of the user for whom recommendations are to be generated.
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (preprocessedData, gameTitles, userGameDetails) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL")

    println("Calculate term frequency and inverse document frequency...")
    val tfidf = timeUtils.time(calculateTFIDF(preprocessedData), "Calculating TF-IDF", "SQL")

    println("Calculate cosine similarity to get similar users...")
    val similarUsers = timeUtils.time(computeCosineSimilarity(tfidf, targetUser), "Finding Similar Users", "SQL")

    println("Calculate final recommendation...")
    timeUtils.time(generateRecommendations(similarUsers, targetUser, gameTitles, userGameDetails), "Generating Recommendations", "SQL")
  }

  /**
   * Preprocesses the input data by normalizing tags and preparing data for TF-IDF.
   *
   * @return A tuple containing:
   *         - A DataFrame with exploded words for users.
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
      .cache()

    val userTagData = userGameDetails
      .withColumn("tag_words", split(col("tags_string"), ","))
      .groupBy("user_id")
      .agg(flatten(collect_list("tag_words")).as("tag_words"))

    val explodedTagsData = userTagData
      .withColumn("word", explode(col("tag_words")))
      .select("user_id", "word")
      .cache()

    val gameTitles = dfGames.select("app_id", "title")

    (explodedTagsData, gameTitles, userGameDetails)
  }

  /**
   * Calculates the Term Frequency-Inverse Document Frequency (TF-IDF) for tags associated with users.
   *
   * @param explodedTagsData A DataFrame containing users and their associated tags.
   * @return A DataFrame containing TF-IDF values for each word-user pair.
   */
  private def calculateTFIDF(explodedTagsData: DataFrame): DataFrame = {
    val wordCountsPerUser = explodedTagsData.groupBy("user_id").agg(count("*").alias("total_word_count"))

    val tf = explodedTagsData.groupBy("user_id", "word")
      .agg(count("*").alias("word_count"))
      .join(wordCountsPerUser, "user_id")
      .withColumn("tf", col("word_count") / col("total_word_count"))

    val totalUsers = explodedTagsData.select("user_id").distinct().count()

    val idf = explodedTagsData.groupBy("word")
      .agg(countDistinct("user_id").alias("user_count"))
      .withColumn("idf", log(lit(totalUsers) / col("user_count")))

    tf.join(idf, "word")
      .withColumn("tfidf", col("tf") * col("idf"))
      .select("user_id", "word", "tfidf")
  }

  /**
   * Computes the cosine similarity between a target user and other users based on TF-IDF vectors.
   *
   * @param tfidf    A DataFrame containing TF-IDF values for tags and users.
   * @param targetUser The ID of the target user.
   * @return A list of user IDs similar to the target user, ordered by similarity.
   */
  private def computeCosineSimilarity(tfidf: DataFrame, targetUser: Int): List[Int] = {
    import spark.implicits._

    val targetUserVector = tfidf.filter($"user_id" === targetUser)
      .select("word", "tfidf")
      .as[(String, Double)]
      .collect()
      .toMap

    val targetNorm = math.sqrt(targetUserVector.values.map(v => v * v).sum)
    val broadcastTargetVector = spark.sparkContext.broadcast(targetUserVector)

    val similarities = tfidf
      .filter($"user_id" =!= targetUser)
      .mapPartitions { partition =>
        val targetVector = broadcastTargetVector.value
        val userScores = scala.collection.mutable.Map[Int, (Double, Double)]()

        partition.foreach { row =>
          val userId = row.getAs[Int]("user_id")
          val word = row.getAs[String]("word")
          val tfidf = row.getAs[Double]("tfidf")

          val targetTfidf = targetVector.getOrElse(word, 0.0)
          val (dotProduct, userNorm) = userScores.getOrElse(userId, (0.0, 0.0))

          userScores(userId) = (
            dotProduct + tfidf * targetTfidf,
            userNorm + tfidf * tfidf
          )
        }

        userScores.iterator.map { case (userId, (dotProduct, userNorm)) =>
          val similarity = if (userNorm == 0.0 || targetNorm == 0.0) 0.0
          else dotProduct / (math.sqrt(userNorm) * targetNorm)
          (userId, similarity)
        }
      }
      .toDF("user_id", "cosine_similarity")

    val topSimilarUsers = similarities
      .orderBy($"cosine_similarity".desc)
      .limit(3)
      .select("user_id")
      .as[Int]
      .collect()
      .toList

    broadcastTargetVector.unpersist()
    topSimilarUsers
  }



  /**
   * Generates the final recommendations for the target user
   *
   * @param similarUserIds  A List[Int] containing the top 3 similar users ids
   * @param targetUser The ID of the target user.
   * @param gameTitles A DataFrame mapping game IDs to their titles.
   * @param userGameDetails - A DataFrame with detailed information about games and users.
   */
  private def generateRecommendations(similarUserIds: List[Int], targetUser: Int, gameTitles: DataFrame, userGameDetails: DataFrame): Unit = {
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
