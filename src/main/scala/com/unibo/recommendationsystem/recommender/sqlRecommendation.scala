package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class sqlRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * Generates recommendations for a specific user.
   *
   * @param targetUser ID of the user for whom recommendations are generated.
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (explodedDF, filteredData, gamesTitles, cleanMerge) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL_FULL")
    println("Calculating term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedDF, filteredData), "Calculating TF-IDF", "SQL_FULL")
    println("Calculating cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "SQL_FULL")
    println("Calculating final recommendation...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, gamesTitles, cleanMerge), "Generating Recommendations", "SQL_FULL")
  }

  /**
   * Preprocesses the data by joining datasets, filtering records, and preparing data for analysis.
   *
   * @return A tuple containing:
   *         - `explodedDF`: DataFrame with individual users and words (tags).
   *         - `filteredData`: DataFrame with aggregated tags for each user.
   *         - `gamesTitles`: DataFrame with game IDs and titles.
   *         - `cleanMerge`: Complete, cleaned, and merged dataset.
   */
  private def preprocessData(): (DataFrame, DataFrame, DataFrame, DataFrame) = {
    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    // Join datasets and remove untagged games
    val merged = selectedRec.join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    // Clean and transform tags
    val cleanMerge = merged
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags")))
      .drop("tags")
      .cache()

    // Create a list of tags for each single user
    val filteredData = cleanMerge
      .withColumn("words", split(col("tagsString"), ",")) // Splits tags by commas
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))

    // Explode tags for calculating TF-IDF
    val explodedDF = filteredData
      .withColumn("word", explode(col("words")))
      .select("user_id", "word")
      .cache()

    val gamesTitles = dataGames.select("app_id", "title")

    (explodedDF, filteredData, gamesTitles, merged)
  }

  /**
   * Calculates TF-IDF values for words in the datasets.
   *
   * @param explodedDF   DataFrame with user-word pairs.
   * @param filteredData DataFrame with aggregated tags for each user.
   * @return DataFrame containing TF-IDF values for each word and user.
   */
  private def calculateTFIDF(explodedDF: DataFrame, filteredData: DataFrame): DataFrame = {
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    // Term Frequency (TF) calculation
    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    // Document Frequency (DF) calculatio
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    // Total number of users (documents of TF-IDF calculation)
    val totalDocs = filteredData.count()

    // Calculate Inverse Document Frequency (IDF)
    val idfDF = dfDF.withColumn("idf", log(lit(totalDocs) / col("document_frequency")))

    // Combine TF and IDF to compute TF-IDF
    val tfidfValues = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    tfidfValues
  }

  /**
   * Calculates cosine similarity between the target user and other users.
   *
   * @param tfidfDF    DataFrame with TF-IDF values for each word and user.
   * @param targetUser ID of the target user.
   * @return List of IDs of the most similar users.
   */
  private def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {
    import spark.implicits._

    // Take the target vector by filtering out target's TF-IDF score
    val targetVector = tfidfDF.filter(col("user_id") === targetUser)
      .select("word", "tf_idf")
      .withColumnRenamed("tf_idf", "target_tfidf")

    // Join the TF-IDF dataset with the target user's vector based on the "word" column
    // Exclude the target user's data and compute the dot product of TF-IDF values
    val joinedDF = tfidfDF
      .join(targetVector, "word")
      .filter(col("user_id") =!= targetUser)
      .withColumn("dot_product", col("tf_idf") * col("target_tfidf"))

    // Aggregate the dot product for each user to compute the numerator of the cosine similarity formula
    val numerator = joinedDF.groupBy("user_id")
      .agg(sum("dot_product").alias("numerator"))

    // Compute the squared TF-IDF values for each user to calculate the norm
    val normDF = tfidfDF.withColumn("squared_tfidf", col("tf_idf") * col("tf_idf"))

    // Aggregate the squared values and take the square root to compute the norm for each user
    val userNorms = normDF.groupBy("user_id")
      .agg(sqrt(sum("squared_tfidf")).alias("user_norm"))

    // Compute the norm for the target user by filtering their data
    val targetNorm = normDF.filter(col("user_id") === targetUser)
      .select(sqrt(sum("squared_tfidf")).alias("target_norm"))
      .as[Double]
      .collect()
      .head // Extract the norm value as a scalar

    // Join the numerator with user norms and calculate the cosine similarity
    // Cosine similarity = numerator / (norm of target user * norm of each user)
    val similarityDF = numerator.join(userNorms, "user_id")
      .withColumn("cosine_similarity", col("numerator") / (col("user_norm") * lit(targetNorm)))

    // Order the users by similarity in descending order, and take the top 3 IDs in a list
    similarityDF.orderBy(desc("cosine_similarity"))
      .limit(3)
      .select("user_id")
      .as[Int]
      .collect()
      .toList
  }


  /**
   * Generates final recommendations by excluding games already played by the target user.
   *
   * @param top3Users   List of IDs of the most similar users.
   * @param targetUser  ID of the target user.
   * @param gamesTitles DataFrame containing game titles.
   * @param cleanMerge  Complete, cleaned, and merged dataset.
   */
  private def generateFinalRecommendations(top3Users: List[Int], targetUser: Int, gamesTitles: DataFrame, cleanMerge: DataFrame): Unit = {
    val gamesByTopUsers = cleanMerge.filter(col("user_id").isin(top3Users: _*)).select("app_id", "user_id")
    val gamesByTargetUser = cleanMerge.filter(col("user_id") === targetUser).select("app_id")

    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")
    val finalRecommendations = recommendedGames.join(gamesTitles.select("app_id", "title"), Seq("app_id"))
      .groupBy("app_id","title")
      .agg(collect_list("user_id").alias("user_ids"))

    finalRecommendations.show(finalRecommendations.count().toInt, truncate = false)
  }
}
