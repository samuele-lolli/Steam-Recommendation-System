package com.unibo.recommendationsystem
import com.unibo.recommendationsystem.utils.schemaUtils
import org.apache.spark.sql.SparkSession

object main {
  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder()
      .appName("Recommendation System")
      .config("spark.master", "local[*]")
      /*.config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
      .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
      .config("spark.executor.memory", "48g") // Allocate 48 GB for each executor
      .config("spark.driver.memory", "8g")    // Allocate 8 GB for the driver
      .config("spark.executor.cores", "4")    // Use 4 cores per executor for parallelism
      .config("spark.default.parallelism", "32") // Set parallelism for transformations
      .config("spark.sql.shuffle.partitions", "32") // Optimize shuffle partitions
      .config("spark.dynamicAllocation.enabled", "true")
      .config("spark.dynamicAllocation.minExecutors", "2")
      .config("spark.dynamicAllocation.maxExecutors", "6")*/
      .getOrCreate()

    val dataPathRec = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\games.csv"
    val metadataPath = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\games_metadata.json"

    val dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(dataPathRec).filter("is_recommended = true")
    val dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(dataPathGames)
    val dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(metadataPath)

    val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    mllibRecommender.recommend(targetUser = 4893896)

    val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    rddRecommender.recommend(targetUser = 4893896)

    val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    sqlRecommender.recommend(targetUser = 4893896)

    sparkLocal.stop()
  }
}