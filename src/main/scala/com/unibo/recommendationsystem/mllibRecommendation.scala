package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class mllibRecommendation (spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  def recommend(targetUser: Int): Unit = {
    // Time the preprocessing of data
    println("Preprocessing data...")
    val (aggregateData, merged) = timeUtils.time(preprocessData(), "Preprocessing Data", "MlLib")
    /*
     Elapsed time for Preprocessing Data:	1906ms (1906329958ns)
     */

    // Time the TF-IDF calculation
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(aggregateData), "Calculating TF-IDF", "MlLib")
    /*
    Elapsed time for Calculating TF-IDF:	214227ms (214227165458ns)
     */

    // Time the similarity computation
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "MlLib")
    /*
    Elapsed time for Getting Similar Users:	8360ms (8360099334ns)
    */
    // Time the recommendation generation
    println("Calculate final recommendation...")
    timeUtils.time(getFinalRecommendations(merged, topUsersSimilarity, targetUser), "Generating Recommendations", "MlLib")
    /*
    Elapsed time for Generating Recommendations:	12281ms (12281858959ns)
     */
    //Total time of execution: 236774ms
  }


  def preprocessData(): (DataFrame, DataFrame) = {

    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    val merged = selectedRec.join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    val cleanMerge = merged.withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags")))
      .drop("tags")
      .persist(StorageLevel.MEMORY_AND_DISK)


    val tokenizedData = cleanMerge.withColumn("words", split(col("tagsString"), ","))

    // Aggregate data
    val aggregateData = tokenizedData.groupBy("user_id").agg(flatten(collect_list("words")).as("words")).persist(StorageLevel.MEMORY_AND_DISK)

    (aggregateData, cleanMerge)
  }

  def calculateTFIDF(aggregateData: DataFrame): DataFrame = {
    // Convert words to feature vectors using HashingTF and IDF
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("hashedFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(aggregateData).persist(StorageLevel.MEMORY_AND_DISK)

    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    rescaledData
  }

  def computeCosineSimilarity(rescaledData: DataFrame, targetUser: Int): DataFrame = {
    import spark.implicits._

    // UDF to convert sparse vectors to dense vectors for cosine similarity
    val denseVector = udf { (v: Vector) => Vectors.dense(v.toArray) }
    val dfWithDenseFeatures = rescaledData.withColumn("dense_features", denseVector(col("features")))

    // Target user features
    val targetUserFeatures = dfWithDenseFeatures.filter($"user_id" === targetUser)
      .select("features").first().getAs[Vector]("features")

    // Cosine similarity calculation
    def cosineSimilarity(targetVector: Vector): UserDefinedFunction = udf { (otherVector: Vector) =>
      val dotProduct = targetVector.dot(otherVector)
      val normA = Vectors.norm(targetVector, 2)
      val normB = Vectors.norm(otherVector, 2)
      dotProduct / (normA * normB)
    }

    val usersSimilarity = dfWithDenseFeatures
      .filter($"user_id" =!= targetUser)
      .withColumn("cosine_sim", cosineSimilarity(targetUserFeatures)(col("features")))
      .select("user_id", "cosine_sim")
      .orderBy($"cosine_sim".desc)
      .limit(3)

    usersSimilarity
  }

  def getFinalRecommendations(merged: DataFrame, usersSimilarity: DataFrame, targetUser: Int) = {
    import spark.implicits._
    val titlesPlayedByTargetUser = merged.filter($"user_id" === targetUser)
      .select("tagsString").distinct().as[String].collect()

    val userIdsToFind = usersSimilarity.select("user_id").as[Int].collect.toSet

    val finalRecommendations = merged.filter(col("user_id").isin(userIdsToFind.toArray: _*) && !col("title").isin(titlesPlayedByTargetUser: _*))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    finalRecommendations.show(finalRecommendations.count.toInt, truncate = false)
  }
}
