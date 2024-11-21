package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import scala.collection.Map

class sqlRecommendationV2 (spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * Generate game recommendations for a specific user.
   *
   * @param targetUser The ID of the user for whom recommendations are generated.
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (explodedDF, filteredData, gamesTitles, cleanMerge) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedDF, filteredData), "Calculating TF-IDF", "SQL")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "SQL")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, gamesTitles, cleanMerge), "Generating Recommendations", "SQL")
  }

  /**
   * Preprocess the input data by cleaning, filtering, and transforming it.
   *
   * @return A tuple containing:
   *         - `explodedDF`: DataFrame with users and individual words (tags).
   *         - `filteredData`: DataFrame with user-wise aggregated tags as arrays.
   *         - `gamesTitles`: DataFrame with game IDs and titles.
   *         - `cleanMerge`: Fully joined and cleaned data set.
   */
  private def preprocessData(): (DataFrame, DataFrame, DataFrame, DataFrame) = {

    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    // Join datasets and filter out records with no tags
    val merged = selectedRec.join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    // Clean and transform tags, persisting the result
    val cleanMerge = merged
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags"))) // Join tags with commas
      .drop("tags")
      .persist(StorageLevel.MEMORY_AND_DISK)

    // Tokenize tags and aggregate them by user
    val filteredData = cleanMerge
      .withColumn("words", split(col("tagsString"), ",")) // Split tags by commas
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))

    // Explode tags for calculating TFIDF
    val explodedDF = filteredData.withColumn("word", explode(col("words"))).select("user_id", "word")
      //.persist(StorageLevel.MEMORY_AND_DISK)

    val gamesTitles = dataGames.select("app_id", "title")

    (explodedDF, filteredData, gamesTitles, merged)
  }

  /**
   * Calculate Term Frequency-Inverse Document Frequency (TF-IDF) values for words in the dataset.
   *
   * @param explodedDF   DataFrame with user-word pairs.
   * @param filteredData DataFrame with user-wise aggregated tags as arrays.
   * @return DataFrame containing TF-IDF scores for each user and word.
   */
  private def calculateTFIDF(explodedDF: DataFrame, filteredData: DataFrame): DataFrame = {
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    // Calculate Term Frequency (TF)
    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    // Calculate Document Frequency (DF)
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    // Total number of documents (users)
    val totalDocs = filteredData.count()

    // Calculate Inverse Document Frequency (IDF)
    dfDF.createOrReplaceTempView("dfDF")
    val idfDF = spark.sql(s"""SELECT word, log($totalDocs / document_frequency) AS idf FROM dfDF""")

    // Combine TF and IDF to compute TF-IDF
    val tfidfDF = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    tfidfDF
  }

  /**
   * Compute the top 3 users with the highest cosine similarity to the target user.
   *
   * @param tfidfDF    DataFrame with TF-IDF scores for each user and word.
   * @param targetUser The ID of the target user.
   * @return List of user IDs with the highest similarity.
   */
  private def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {
    // Define helper functions for cosine similarity calculation
    def calculateCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double], dotProductFunc: (Map[String, Double], Map[String, Double]) => Double, magnitudeFunc: Map[String, Double] => Double): Double = {
      val magnitude1 = magnitudeFunc(vector1)
      val magnitude2 = magnitudeFunc(vector2)
      if (magnitude1 == 0.0 || magnitude2 == 0.0) 0.0 else dotProductFunc(vector1, vector2) / (magnitude1 * magnitude2)
    }

    val dotProduct = (v1: Map[String, Double], v2: Map[String, Double]) => {
      v1.foldLeft(0.0) { case (acc, (key, value)) => acc + v2.getOrElse(key, 0.0) * value }
    }

    val magnitude = (vector: Map[String, Double]) => math.sqrt(vector.values.map(value => value * value).sum)

    def rowToTfIdfMap(row: Row): Map[String, Double] = {
      row.getAs[Seq[Row]]("tags").map(tag => tag.getString(0) -> tag.getDouble(1)).toMap
    }

    // Extract target user vector
    val targetUserData = tfidfDF.filter(col("user_id") === targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))

    val targetUserVector = targetUserData.collect().headOption.map(rowToTfIdfMap).getOrElse(Map.empty[String, Double])

    // Compute similarity for other users
    val otherUsersData = tfidfDF.filter(col("user_id") =!= targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))

    import org.apache.spark.sql.Encoders
    implicit val tupleEncoder: Encoder[(Int, Double)] = Encoders.product[(Int, Double)]

    val otherUsersWithSimilarity = otherUsersData.map { row =>
      val userId = row.getAs[Int]("user_id")
      val userVector = rowToTfIdfMap(row)
      val cosineSimilarity = calculateCosineSimilarity(targetUserVector, userVector, dotProduct, magnitude)
      (userId, cosineSimilarity)
    }.toDF("user_id", "cosine_similarity")

    val top3Users = otherUsersWithSimilarity.orderBy(desc("cosine_similarity")).limit(3)
    top3Users.select("user_id").collect().map(row => row.getAs[Int]("user_id")).toList
  }

  /**
   * Generate final recommendations based on similar users and exclude games already played by the target user.
   *
   * @param top3Users  List of top similar user IDs.
   * @param targetUser The ID of the target user.
   * @param gamesTitles DataFrame containing game IDs and titles.
   * @param cleanMerge Fully joined and cleaned data set.
   */
  def generateFinalRecommendations(top3Users: List[Int], targetUser: Int, gamesTitles: DataFrame, cleanMerge: DataFrame) = {

    val gamesByTopUsers = cleanMerge.filter(col("user_id").isin(top3Users: _*)) // Use : _* to expand the list
      .select("app_id", "user_id")

    val gamesByTargetUser = cleanMerge.filter(col("user_id") === targetUser)
      .select("app_id")

    //Exclude the games played by the target user from the games played by the similar users
    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")

    //Join with dfGames to get the titles of the recommended games
    val finalRecommendations = recommendedGames
      .join(gamesTitles.select("app_id", "title"), Seq("app_id"))
      .select("title", "user_id")

    // Show the resulting DataFrame with titles and users
    val groupedRecommendations = finalRecommendations
      .groupBy("title")
      .agg(collect_list("user_id").alias("user_ids")) // Aggregate user_ids for each title
      .select("title", "user_ids") // Select only the title and aggregated user_ids

    groupedRecommendations.show(groupedRecommendations.count.toInt, truncate = false)
  }
}
