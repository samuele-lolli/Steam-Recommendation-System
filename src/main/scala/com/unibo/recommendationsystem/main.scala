package com.unibo.recommendationsystem
import com.unibo.recommendationsystem.utils.schemaUtils
import org.apache.spark.sql.SparkSession

object main {
  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder().appName("Recommendation System").config("spark.master", "local[*]").getOrCreate()

    val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"
    val metadataPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games_metadata.json"

    val dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(dataPathRec).filter("is_recommended = true")
    val dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(dataPathGames)
    val dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(metadataPath)

    //val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    //mllibRecommender.recommend(targetUser = 4893896)

   // val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
   // rddRecommender.recommend(targetUser = 4893896)

    val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    sqlRecommender.recommend(targetUser = 4893896)

    sparkLocal.stop()
  }
}