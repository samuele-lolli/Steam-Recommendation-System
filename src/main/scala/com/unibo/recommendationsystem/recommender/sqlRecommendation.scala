package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class sqlRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * Main method for generating recommendations for a target user.
   *
   * @param targetUser The user_id of the target user for whom we want to generate recommendations.
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    // Preprocess the data and return exploded DataFrame, game titles, and user-games data
    val (explodedDF, gamesTitles, userGamesData) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL_FULL")

    println("Calculating term frequency and inverse document frequency...")
    // Calculate the TF-IDF values based on the exploded data
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedDF), "Calculating TF-IDF", "SQL_FULL")

    println("Calculating cosine similarity to get similar users...")
    // Compute the cosine similarity to identify similar users to the target user
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "SQL_FULL")

    println("Calculating final recommendation...")
    // Generate the final recommendations based on the most similar users
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, gamesTitles, userGamesData), "Generating Recommendations", "SQL_FULL")
  }

  /**
   * Preprocess the data to explode the tags into individual words for each user.
   *
   * @return A tuple containing the exploded DataFrame (with user_id and word columns),
   *         game titles DataFrame, and the user-games data DataFrame.
   */
  private def preprocessData(): (DataFrame, DataFrame, DataFrame) = {
    // Select the relevant columns from the recommendations and games data
    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    // Join data to combine user, game, and metadata, and filter games with tags
    val userGamesData = selectedRec
      .join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    // Clean and transform tags (remove spaces and lower case)
    val userGamePairs = userGamesData
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))) )
      .withColumn("tagsString", concat_ws(",", col("tags")))
      .drop("tags")

    // Create a list of words (tags) for each user
    val filteredData = userGamePairs
      .withColumn("words", split(col("tagsString"), ","))
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))

    // Explode the list of words so we can calculate the term frequency (TF-IDF)
    val explodedDF = filteredData
      .withColumn("word", explode(col("words")))
      .select("user_id", "word")
      .persist(StorageLevel.MEMORY_AND_DISK)

    val gamesTitles = dataGames.select("app_id", "title")

    (explodedDF, gamesTitles, userGamesData)
  }

  /**
   * Calculate the Term Frequency-Inverse Document Frequency (TF-IDF) for each user and word.
   *
   * @param explodedDF The exploded DataFrame containing user_id and word columns.
   * @return A DataFrame with user_id, word, and tf-idf values.
   */
  private def calculateTFIDF(explodedDF: DataFrame): DataFrame = {
    // Count total words per user
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    // Calculate term frequency (TF)
    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    // Calculate document frequency (DF) for each word
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    // Calculate inverse document frequency (IDF)
    val totalDocs = explodedDF.select("user_id").distinct().count()
    val idfDF = dfDF.withColumn("idf", log(lit(totalDocs) / col("document_frequency")))

    // Join TF and IDF to calculate TF-IDF
    val tfidfValues = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    tfidfValues
  }

  /**
   * Compute the cosine similarity between the target user and all other users.
   *
   * @param tfidfDF The DataFrame containing the TF-IDF values for each user and word.
   * @param targetUser The user_id of the target user for similarity calculation.
   * @return A list of user_ids with the highest cosine similarity to the target user.
   */
  private def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {
    import spark.implicits._

    // Filter out the target user's data and create a map of word -> TF-IDF values for the target user
    val targetVector = tfidfDF.filter($"user_id" === targetUser)
      .select("word", "tf_idf")
      .withColumnRenamed("tf_idf", "target_tfidf")

    // Broadcast the target user's vector to all workers
    val targetBroadcast = spark.sparkContext.broadcast(
      targetVector.as[(String, Double)].collect().toMap
    )

    // Compute cosine similarity for each user by calculating the numerator and norm
    val similarityDF = tfidfDF
      .filter($"user_id" =!= targetUser) // Exclude the target user from the similarity calculation
      .mapPartitions { partition =>
        val targetMap = targetBroadcast.value
        val userScores = scala.collection.mutable.Map[Int, (Double, Double)]() // (numerator, norm)

        partition.foreach { row =>
          val userId = row.getAs[Int]("user_id")
          val word = row.getAs[String]("word")
          val tfidf = row.getAs[Double]("tf_idf")
          val targetTfidf = targetMap.getOrElse(word, 0.0)

          // Update the numerator and norm for each user
          val (numerator, norm) = userScores.getOrElse(userId, (0.0, 0.0))
          userScores(userId) = (numerator + tfidf * targetTfidf, norm + tfidf * tfidf)
        }

        userScores.iterator.map { case (userId, (numerator, norm)) =>
          (userId, numerator, Math.sqrt(norm))
        }
      }
      .toDF("user_id", "numerator", "user_norm")

    // Calculate the norm for the target user
    val targetNorm = Math.sqrt(targetBroadcast.value.values.map(v => v * v).sum)

    // Compute cosine similarity and get the top 3 most similar users
    val topUsers = similarityDF
      .withColumn("cosine_similarity", $"numerator" / ($"user_norm" * lit(targetNorm)))
      .orderBy(desc("cosine_similarity"))
      .limit(3)
      .select("user_id")
      .as[Int]
      .collect()

    // Unpersist the broadcast variable to free up memory
    targetBroadcast.unpersist()
    topUsers.toList
  }

  /**
   * Generate final game recommendations based on the most similar users.
   *
   * @param top3Users The list of user_ids with the highest cosine similarity to the target user.
   * @param targetUser The user_id of the target user.
   * @param gamesTitles The DataFrame containing game titles.
   * @param userGamesData The DataFrame containing the games played by each user.
   */
  private def generateFinalRecommendations(top3Users: List[Int], targetUser: Int, gamesTitles: DataFrame, userGamesData: DataFrame): Unit = {
    // Filter games played by the top 3 similar users
    val gamesByTopUsers = userGamesData.filter(col("user_id").isin(top3Users: _*)).select("app_id", "user_id")
    val gamesByTargetUser = userGamesData.filter(col("user_id") === targetUser).select("app_id")

    // Get the recommended games by finding games played by similar users but not by the target user
    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")
    val finalRecommendations = recommendedGames.join(gamesTitles.select("app_id", "title"), Seq("app_id"))
      .groupBy("app_id","title")
      .agg(collect_list("user_id").alias("user_ids"))

    finalRecommendations.show(finalRecommendations.count().toInt, truncate = false)
  }
}
