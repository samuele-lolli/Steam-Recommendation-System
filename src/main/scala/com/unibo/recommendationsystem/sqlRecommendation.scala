package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import scala.collection.Map

class sqlRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (explodedDF, filteredData, gamesTitles, cleanMerge) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL")
    println("Calculating term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedDF, filteredData), "Calculating TF-IDF", "SQL")
    println("Calculating cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "SQL")
    println("Calculating final recommendation...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, gamesTitles, cleanMerge), "Generating Recommendations", "SQL")
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
      .cache()

    val filteredData = cleanMerge
      .withColumn("words", split(col("tagsString"), ",")) // Split on commas, preserving multi-word tags
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))

    val explodedDF = filteredData
      .withColumn("word", explode(col("words")))
      .select("user_id", "word")
      .cache()

    val gamesTitles = dataGames.select("app_id", "title")

    (explodedDF, filteredData, gamesTitles, merged)
  }

  private def calculateTFIDF(explodedDF: DataFrame, filteredData: DataFrame): DataFrame = {
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    val totalDocs = filteredData.count()

    val idfDF = dfDF.withColumn("idf", log(lit(totalDocs) / col("document_frequency")))

    tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")
  }

  private def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {
    def calculateCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
      val dotProduct = vector1.keys.map(k => vector1(k) * vector2.getOrElse(k, 0.0)).sum
      val magnitude1 = math.sqrt(vector1.values.map(v => v * v).sum)
      val magnitude2 = math.sqrt(vector2.values.map(v => v * v).sum)
      if (magnitude1 == 0.0 || magnitude2 == 0.0) 0.0 else dotProduct / (magnitude1 * magnitude2)
    }

    def rowToTfIdfMap(row: Row): Map[String, Double] = row.getAs[Seq[Row]]("tags").map(tag => tag.getString(0) -> tag.getDouble(1)).toMap

    val targetUserVector = tfidfDF.filter(col("user_id") === targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))
      .collect()
      .headOption
      .map(rowToTfIdfMap)
      .getOrElse(Map.empty[String, Double])

    val otherUsersData = tfidfDF.filter(col("user_id") =!= targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))

    import spark.implicits._
    val otherUsersWithSimilarity = otherUsersData.map { row =>
      val userId = row.getAs[Int]("user_id")
      val userVector = rowToTfIdfMap(row)
      val cosineSimilarity = calculateCosineSimilarity(targetUserVector, userVector)
      (userId, cosineSimilarity)
    }.toDF("user_id", "cosine_similarity")

    val debug = otherUsersWithSimilarity.orderBy(desc("cosine_similarity"))
      .limit(3)
      .select("user_id")
      .as[Int]
      .collect()
      .toList

    debug.foreach(println)
    println("top 10 users sql")

    debug
  }

  private def generateFinalRecommendations(top3Users: List[Int], targetUser: Int, gamesTitles: DataFrame, cleanMerge: DataFrame): Unit = {
    val gamesByTopUsers = cleanMerge.filter(col("user_id").isin(top3Users: _*)).select("app_id", "user_id")
    val gamesByTargetUser = cleanMerge.filter(col("user_id") === targetUser).select("app_id")

    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")
    val finalRecommendations = recommendedGames.join(gamesTitles.select("app_id", "title"), Seq("app_id"))
      .groupBy("title")
      .agg(collect_list("user_id").alias("user_ids"))

    finalRecommendations.show(finalRecommendations.count().toInt, truncate = false)
  }
}
