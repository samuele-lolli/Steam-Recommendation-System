package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.recommender.{mllibRecommendation, parRecommendation, rddRecommendation, seqRecommendation, sqlRecommendation}
import com.unibo.recommendationsystem.utils.dataUtils.createCustomDatasets
import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.{DataFrame, SparkSession}

object main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Recommendation System")
      .config("spark.master", "local[*]")
      .getOrCreate()
    spark.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    val basePath = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\"

    val (targetUser, mode) = createCustomDatasets(spark, basePath)

    if (mode == "full") {
      val dfRecFull = spark.read
        .format("csv")
        .option("header", "true")
        .schema(schemaUtils.recSchema)
        .load(basePath + "recommendations.csv")
        .filter("is_recommended = true")

      val dfGamesFull = spark.read
        .format("csv")
        .option("header", "true")
        .schema(schemaUtils.gamesSchema)
        .load(basePath + "games.csv")

      val dfMetadataFull = spark.read
        .format("json")
        .schema(schemaUtils.metadataSchema)
        .load(basePath + "games_metadata.json")

      runRecommendersFull(spark, dfRecFull, dfGamesFull, dfMetadataFull, targetUser)
    } else {
      val dfRec = spark.read
        .format("csv")
        .option("header", "true")
        .schema(schemaUtils.recSchema)
        .load(basePath + "filteredDataset/recommendations_f.csv")

      val dfGames = spark.read
        .format("csv")
        .option("header", "true")
        .schema(schemaUtils.gamesSchema)
        .load(basePath + "filteredDataset/games_f.csv")

      val dfMetadata = spark.read
        .format("json")
        .schema(schemaUtils.metadataSchema)
        .load(basePath + "filteredDataset/games_metadata_f.json")

      runRecommendersFiltered(basePath, spark, dfRec, dfGames, dfMetadata, targetUser)
    }
    spark.stop()
  }

  private def runRecommendersFull(spark: SparkSession, dfRec: DataFrame, dfGames: DataFrame, dfMetadata: DataFrame, targetUser: Int): Unit = {
    val mllibRecommender = new mllibRecommendation(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MlLib")

    val rddRecommender = new rddRecommendation(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")

    val sqlRecommender = new sqlRecommendation(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL", "SQL")
  }

  private def runRecommendersFiltered(basePath: String, spark: SparkSession, dfRec: DataFrame, dfGames: DataFrame, dfMetadata: DataFrame, targetUser: Int): Unit = {
    val seqRecommender = new seqRecommendation(
      basePath + "filteredDataset/recommendations_f.csv",
      basePath + "filteredDataset/games_f.csv",
      basePath + "filteredDataset/games_metadata_f.json"
    )
    timeUtils.time(seqRecommender.recommend(targetUser), "Total time execution sequential", "Seq")

    val parRecommender = new parRecommendation(
      basePath + "filteredDataset/recommendations_f.csv",
      basePath + "filteredDataset/games_f.csv",
      basePath + "filteredDataset/games_metadata_f.json"
    )
    timeUtils.time(parRecommender.recommend(targetUser), "Total time execution parallel", "Par")

    val mllibRecommender = new mllibRecommendation(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MlLib")

    val rddRecommender = new rddRecommendation(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")

    val sqlRecommender = new sqlRecommendation(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL", "SQL")
  }
}
