package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

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

    val tfidfValues = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    tfidfValues
  }

  private def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {
    import spark.implicits._
    // Vettore TF-IDF dell'utente target
    val targetVector = tfidfDF.filter(col("user_id") === targetUser)
      .select("word", "tf_idf")
      .withColumnRenamed("tf_idf", "target_tfidf")

    // Unisci tfidfDF con targetVector sui campi "word" per calcolare il prodotto scalare
    val joinedDF = tfidfDF
      .join(targetVector, "word")
      .filter(col("user_id") =!= targetUser)
      .withColumn("dot_product", col("tf_idf") * col("target_tfidf"))

    // Calcola il numeratore (somma dei prodotti)
    val numerator = joinedDF.groupBy("user_id")
      .agg(sum("dot_product").alias("numerator"))

    // Calcola la norma per ciascun utente e per il target
    val normDF = tfidfDF.withColumn("squared_tfidf", col("tf_idf") * col("tf_idf"))
    val userNorms = normDF.groupBy("user_id")
      .agg(sqrt(sum("squared_tfidf")).alias("user_norm"))
    val targetNorm = normDF.filter(col("user_id") === targetUser)
      .select(sqrt(sum("squared_tfidf")).alias("target_norm"))
      .as[Double]
      .collect()
      .head

    // Calcola la similarità usando il prodotto scalare e le norme
    val similarityDF = numerator.join(userNorms, "user_id")
      .withColumn("cosine_similarity", col("numerator") / (col("user_norm") * lit(targetNorm)))

    // Ordina gli utenti per similarità e prendi i primi 3
    similarityDF.orderBy(desc("cosine_similarity"))
      .limit(3)
      .select("user_id")
      .as[Int]
      .collect()
      .toList
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
