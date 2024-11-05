package com.unibo.recommendationsystem

import org.apache.spark.sql.SparkSession

object main {
  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder().appName("Recommendation System").config("spark.master", "local[*]").getOrCreate()

    val dataPathRec = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\games.csv"
    val metadataPath = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\games_metadata.json"

    val sqlRecommender = new sqlRecommendation(sparkLocal, dataPathRec, dataPathGames, metadataPath)
    sqlRecommender.recommend(targetUser = 4893896)

//    val mllibRecommender = new mllibRecommendation(sparkLocal, dataPathRec, dataPathGames, metadataPath)
//    mllibRecommender.recommend(targetUser = 4893896)
//
//    val rddRecommender = new rddRecommendation(sparkLocal, dataPathRec, dataPathGames, metadataPath)
//    rddRecommender.recommend(targetUser = 4893896)

    sparkLocal.stop()
  }
}