package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.recommender.{mllibRecommendation, rddRecommendation, sqlRecommendation, sqlRecommendationV2}
import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.SparkSession

object distributedMain {

  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder()
      .appName("RecommendationSystem")
      .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
      .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
      .config("spark.executor.memory", "48g") // Allocate 48 GB for each executor
      .config("spark.driver.memory", "8g")    // Allocate 8 GB for the driver
      .config("spark.executor.cores", "4")    // Use 4 cores per executor for parallelism
      .config("spark.default.parallelism", "32") // Set parallelism for transformations
      .config("spark.sql.shuffle.partitions", "32") // Optimize shuffle partitions
      .config("spark.dynamicAllocation.enabled", "true")
      .config("spark.dynamicAllocation.minExecutors", "2")
      .config("spark.dynamicAllocation.maxExecutors", "6")

/*
  .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
  .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
  .config("spark.executor.memory", "48g") // 48 GB for executors
  .config("spark.executor.memoryOverhead", "6g") // Overhead memory
  .config("spark.driver.memory", "16g") // Increased driver memory
  .config("spark.executor.cores", "4") // 4 cores per executor
  .config("spark.default.parallelism", "48") // 2× total cores
  .config("spark.sql.shuffle.partitions", "48") // Matches parallelism
  .config("spark.dynamicAllocation.enabled", "true")
  .config("spark.dynamicAllocation.minExecutors", "2")
  .config("spark.dynamicAllocation.maxExecutors", "6")
  .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  .config("spark.sql.adaptive.enabled", "true") // Enable adaptive execution
*/

  .getOrCreate()

val basePath = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/"
timeUtils.setLogFilePath(basePath+"result.txt")
val targetUser = 4893896

val dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath + "recommendations.csv").filter("is_recommended = true")
val dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath + "games.csv")
val dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "games_metadata.json")

val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MlLib")

val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")

val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL", "SQL")

val sqlV2Recommender = new sqlRecommendationV2(sparkLocal, dfRec, dfGames, dfMetadata)
timeUtils.time(sqlV2Recommender.recommend(targetUser), "Total time execution SQLV2", "SQLV2")
}
}

