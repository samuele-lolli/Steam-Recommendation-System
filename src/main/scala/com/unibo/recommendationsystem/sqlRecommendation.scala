package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.Map

class sqlRecommendation (spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    //Elapsed time for Preprocessing Data:	1495ms (1495704000ns)
    val (explodedDF, filteredData, gamesTitles, cleanMerge) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedDF, filteredData), "Calculating TF-IDF", "SQL")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "SQL")
    println("Calculate final recommendation...")
    timeUtils.time(getFinalRecommendations(topUsersSimilarity, targetUser, gamesTitles, cleanMerge), "Generating Recommendations", "SQL")
  }

  private def preprocessData(): (DataFrame, DataFrame, DataFrame, DataFrame) = {

    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    val merged = selectedRec.join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    val cleanMerge = merged
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags"))) // Join tags with commas
      .drop("tags")
      .persist(StorageLevel.MEMORY_AND_DISK)

    // Tokenize by splitting on commas to maintain multi-word tags as single elements
    val filteredData = cleanMerge
      .withColumn("words", split(col("tagsString"), ",")) // Split on commas, preserving multi-word tags
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))


    // Explode aggregated data for TF-IDF calculation
    val explodedDF = filteredData.withColumn("word", explode(col("words"))).select("user_id", "word")
      .persist(StorageLevel.MEMORY_AND_DISK)

    val gamesTitles = dataGames.select("app_id", "title")

    (explodedDF, filteredData, gamesTitles, merged)
  }

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

    // Calculate total number of users
    val totalDocs = filteredData.count()

    // Calculate IDF for each word using SQL
    dfDF.createOrReplaceTempView("dfDF")
    val idfDF = spark.sql(s"""SELECT word, log($totalDocs / document_frequency) AS idf FROM dfDF""")

    // Join TF and IDF to get TF-IDF
    val tfidfDF = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    tfidfDF
  }

  def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {
    // Define the cosine similarity function
    def calculateCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double], dotProductFunc: (Map[String, Double], Map[String, Double]) => Double, magnitudeFunc: Map[String, Double] => Double): Double = {
      val magnitude1 = magnitudeFunc(vector1)
      val magnitude2 = magnitudeFunc(vector2)
      if (magnitude1 == 0.0 || magnitude2 == 0.0) 0.0 // Avoid division by zero
      else dotProductFunc(vector1, vector2) / (magnitude1 * magnitude2)
    }

    // Define the dot product and magnitude functions
    val dotProduct = (v1: Map[String, Double], v2: Map[String, Double]) => {
      v1.foldLeft(0.0) { case (acc, (key, value)) =>
        acc + v2.getOrElse(key, 0.0) * value
      }
    }

    val magnitude = (vector: Map[String, Double]) => {
      math.sqrt(vector.values.map(value => value * value).sum)
    }

    // Convert DataFrame rows to TF-IDF maps
    def rowToTfIdfMap(row: Row): Map[String, Double] = {
      row.getAs[Seq[Row]]("tags").map(tag => tag.getString(0) -> tag.getDouble(1)).toMap
    }

    // Step 1: Extract TF-IDF vector for the target user
    val targetUserData = tfidfDF.filter(col("user_id") === targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))

    val targetUserVector = targetUserData.collect().headOption.map(rowToTfIdfMap).getOrElse(Map.empty[String, Double])

    // Step 2: Calculate cosine similarity with other users
    val otherUsersData = tfidfDF.filter(col("user_id") =!= targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))

    import org.apache.spark.sql.Encoders

    // Define an implicit encoder for (Int, Double)
    implicit val tupleEncoder: Encoder[(Int, Double)] = Encoders.product[(Int, Double)]
    // Now you can use DataFrame map with the encoder
    val otherUsersWithSimilarity = otherUsersData.map { row =>
      val userId = row.getAs[Int]("user_id")
      val userVector = rowToTfIdfMap(row)
      val cosineSimilarity = calculateCosineSimilarity(targetUserVector, userVector, dotProduct, magnitude)
      (userId, cosineSimilarity)
    }.toDF("user_id", "cosine_similarity")


    // Step 3: Get the top 3 users with highest cosine similarity
    val top3Users = otherUsersWithSimilarity.orderBy(desc("cosine_similarity")).limit(3)

    val topSimilarUsers = top3Users.select("user_id").collect().map(row => row.getAs[Int]("user_id")).toList

    topSimilarUsers
  }

  /*

  Top 3 users with highest cosine similarity:
userId: 8971360, cosine similarity: 0.8591424719530733
userId: 11277999, cosine similarity: 0.8436706750570966
userId: 9911449, cosine similarity: 0.8421752054744202
Elapsed time for Getting Similar Users:	533863ms (533863187600ns)
  */

  def getFinalRecommendations(top3Users: List[Int], targetUser: Int, gamesTitles: DataFrame, cleanMerge: DataFrame) = {

    val gamesByTopUsers = cleanMerge.filter(col("user_id").isin(top3Users: _*)) // Use : _* to expand the list
      .select("app_id", "user_id")

    // Step 3: Fetch the games played by the target user
    val gamesByTargetUser = cleanMerge.filter(col("user_id") === targetUser)
      .select("app_id")

    // Step 4: Exclude the games played by the target user from the games played by the similar users
    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")

    // Step 5: Join with dfGames to get the titles of the recommended games
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