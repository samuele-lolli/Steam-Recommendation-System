package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.recommender.{mllibRecommendation, parRecommendation, rddRecommendation, seqRecommendation, sqlRecommendation}
import com.unibo.recommendationsystem.utils.dataUtils.createCustomDatasets
import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.SparkSession

object main {
  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder()
      .appName("Recommendation System")
      .config("spark.master", "local[*]")
      .getOrCreate()
    sparkLocal.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    val basePath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/"

    //4893896
    val (targetUser, mode) = createCustomDatasets(sparkLocal)

    if(mode.equals("full")){
      val dfRecFull = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath + "recommendations.csv").filter("is_recommended = true")
      val dfGamesFull= sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath + "games.csv")
      val dfMetadataFull = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "games_metadata.json")

      val mllibRecommender = new mllibRecommendation(sparkLocal, dfRecFull, dfGamesFull, dfMetadataFull)
      timeUtils.time(mllibRecommender.recommend(4893896), "Total time execution MlLib", "MlLib")

      val rddRecommender = new rddRecommendation(sparkLocal, dfRecFull, dfGamesFull, dfMetadataFull)
      timeUtils.time(rddRecommender.recommend(4893896), "Total time execution RDD", "RDD")

      val sqlRecommender = new sqlRecommendation(sparkLocal, dfRecFull, dfGamesFull, dfMetadataFull)
        timeUtils.time(sqlRecommender.recommend(4893896), "Total time execution SQL", "SQL")

    } else {
      val dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath + "filteredDataset/recommendations_f.csv")
      val dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath + "filteredDataset/games_f.csv")
      val dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "filteredDataset/games_metadata_f.json")

      val seqRecommender = new seqRecommendation(basePath+"filteredDataset/recommendations_f.csv", basePath+"filteredDataset/games_f.csv", basePath+"filteredDataset/games_metadata_f.json")
       timeUtils.time(seqRecommender.recommend(targetUser), "Total time execution sequential", "Seq")

      val parRecommender = new parRecommendation(basePath+"filteredDataset/recommendations_f.csv", basePath+"filteredDataset/games_f.csv", basePath+"filteredDataset/games_metadata_f.json")
       timeUtils.time(parRecommender.recommend(targetUser), "Total time execution parallel", "Par")


      val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MlLib")



      val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")


      val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL", "SQL")


    }



    sparkLocal.stop()
  }
}