package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.recommender.{mllibRecommendation, parRecommendation, rddRecommendation, seqRecommendation, sqlRecommendation, sqlRecommendationV2}
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

    /* Prompt the user to either use the full dataset or create a custom one. */
    val (targetUser, mode) = createCustomDatasets(spark, basePath)

    /* If the user selected the full dataset, load the full datasets. */
    if (mode == "full") {
      /* Load the full recommendations data, filtering for recommended items. */
      val dfRecFull = spark.read
        .format("csv")
        .option("header", "true")
        .schema(schemaUtils.recSchema)
        .load(basePath + "recommendations.csv")
        .filter("is_recommended = true")

      /* Load the full games dataset. */
      val dfGamesFull = spark.read
        .format("csv")
        .option("header", "true")
        .schema(schemaUtils.gamesSchema)
        .load(basePath + "games.csv")

      /* Load the full metadata dataset. */
      val dfMetadataFull = spark.read
        .format("json")
        .schema(schemaUtils.metadataSchema)
        .load(basePath + "games_metadata.json")

      /* Run recommendation algorithms on the full dataset. */
      runRecommendersFull(spark, dfRecFull, dfGamesFull, dfMetadataFull, targetUser)
    } else {
      /* If the user selected a custom dataset, load the filtered datasets. */
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

      /* Run recommendation algorithms on the filtered dataset. */
      runRecommendersFiltered(basePath, spark, dfRec, dfGames, dfMetadata, targetUser)
    }
    spark.stop()
  }

  /**
   * Runs the recommendation algorithms on the full dataset.
   *
   * @param spark Spark session.
   * @param dfRec DataFrame containing the recommendations.
   * @param dfGames DataFrame containing the game details.
   * @param dfMetadata DataFrame containing the game metadata.
   * @param targetUser The user ID for which recommendations are generated.
   */
  private def runRecommendersFull(spark: SparkSession, dfRec: DataFrame, dfGames: DataFrame, dfMetadata: DataFrame, targetUser: Int): Unit = {
//    /* Initialize and run the MLlib recommender algorithm. */
//    val mllibRecommender = new mllibRecommendation(spark, dfRec, dfGames, dfMetadata)
//    timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MlLib")
//
//    /* Initialize and run the RDD-based recommender algorithm. */
//    val rddRecommender = new rddRecommendation(spark, dfRec, dfGames, dfMetadata)
//    timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")

    /* Initialize and run the SQL-hybrid-based recommender algorithm. */
    val sqlRecommenderV2 = new sqlRecommendationV2(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(sqlRecommenderV2.recommend(targetUser), "Total time execution SQL_HYBRID", "SQL_HYBRID")

    /* Initialize and run the SQL-full-based recommender algorithm. */
    val sqlRecommender = new sqlRecommendation(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL_FULL", "SQL_FULL")
  }

  /**
   * Runs the recommendation algorithms on the filtered dataset.
   *
   * @param basePath Base path to the filtered dataset.
   * @param spark Spark session.
   * @param dfRec DataFrame containing the filtered recommendations.
   * @param dfGames DataFrame containing the filtered game details.
   * @param dfMetadata DataFrame containing the filtered game metadata.
   * @param targetUser The user ID for which recommendations are generated.
   */
  private def runRecommendersFiltered(basePath: String, spark: SparkSession, dfRec: DataFrame, dfGames: DataFrame, dfMetadata: DataFrame, targetUser: Int): Unit = {
    /* Initialize and run the sequential recommender algorithm. */
//    val seqRecommender = new seqRecommendation(
//      basePath + "filteredDataset/recommendations_f.csv",
//      basePath + "filteredDataset/games_f.csv",
//      basePath + "filteredDataset/games_metadata_f.json"
//    )
//    timeUtils.time(seqRecommender.recommend(targetUser), "Total time execution sequential", "Seq")
//
//    /* Initialize and run the parallel recommender algorithm. */
//    val parRecommender = new parRecommendation(
//      basePath + "filteredDataset/recommendations_f.csv",
//      basePath + "filteredDataset/games_f.csv",
//      basePath + "filteredDataset/games_metadata_f.json"
//    )
//    timeUtils.time(parRecommender.recommend(targetUser), "Total time execution parallel", "Par")
//
//    /* MLlib-based recommender. */
//    val mllibRecommender = new mllibRecommendation(spark, dfRec, dfGames, dfMetadata)
//    timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MlLib")
//
//    /* RDD-based recommender. */
//    val rddRecommender = new rddRecommendation(spark, dfRec, dfGames, dfMetadata)
//    timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")
//
//    /* SQL-full-based recommender. */
//    val sqlRecommender = new sqlRecommendation(spark, dfRec, dfGames, dfMetadata)
//    timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL_FULL", "SQL_FULL")

    /* SQL-hybrid-based recommender. */
    val sqlRecommenderV2 = new sqlRecommendationV2(spark, dfRec, dfGames, dfMetadata)
    timeUtils.time(sqlRecommenderV2.recommend(targetUser), "Total time execution SQL_HYBRID", "SQL_HYBRID")
  }
}
