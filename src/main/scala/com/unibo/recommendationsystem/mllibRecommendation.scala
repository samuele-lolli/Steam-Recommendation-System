package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class mllibRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (aggregateData, merged) = timeUtils.time(preprocessData(), "Preprocessing Data", "MlLib")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(aggregateData), "Calculating TF-IDF", "MlLib")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "MlLib")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(merged, topUsersSimilarity, targetUser), "Generating Recommendations", "MlLib")
  }

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

  private def calculateTFIDF(aggregateData: DataFrame): DataFrame = {
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("hashedFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(aggregateData).cache()

    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    idfModel.transform(featurizedData)
  }

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

  private def generateFinalRecommendations(merged: DataFrame, usersSimilarity: DataFrame, targetUser: Int): Unit = {
    import spark.implicits._

    val titlesPlayedByTargetUser = merged
      .filter($"user_id" === targetUser)
      .select("tagsString")
      .distinct()
      .as[String]
      .collect()

    val userIdsToFind = usersSimilarity
      .select("user_id")
      .as[Int]
      .collect()
      .toSet

    val finalRecommendations = merged
      .filter(col("user_id").isin(userIdsToFind.toSeq: _*) && !col("title").isin(titlesPlayedByTargetUser: _*))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    finalRecommendations.show(finalRecommendations.count().toInt ,truncate = false)
  }
}