package com.unibo.recommendationsystem
import com.unibo.recommendationsystem.utils.schemaUtils
import org.apache.spark.sql.SparkSession

object main {
  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder().appName("Recommendation System")//.config("spark.master", "local[*]")
      .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
      .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
      .getOrCreate()

    val dataPathRec = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/recommendations.csv"
    val dataPathGames = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/games.csv"
    val metadataPath = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/games_metadata.json"

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