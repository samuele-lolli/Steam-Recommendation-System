package com.unibo.recommendationsystem
import com.unibo.recommendationsystem.recommender.{parRecommendation, seqRecommendation}
import com.unibo.recommendationsystem.utils.dataUtils.{createCustomDatasets, filterAppIds, filterUsersWithReviews, saveFilteredDataset}
import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, concat_ws}
import org.apache.spark.sql.functions._

object main {
  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder()
      .appName("Recommendation System")
      .config("spark.master", "local[*]")
      .getOrCreate()
    sparkLocal.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    var basePath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/"

    var dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath + "recommendations.csv").filter("is_recommended = true").sample(withReplacement = false, fraction = 0.35)
    var dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath + "games.csv")
    var dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "games_metadata.json")
    val dfUsers = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.usersSchema).load(basePath + "users.csv")

    //4893896
    val targetUser = createCustomDatasets(sparkLocal, dfRec, dfGames, dfMetadata, dfUsers)

      basePath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/filteredDataset/"

      val seqRecommender = new seqRecommendation(basePath+"recommendations_f.csv", basePath+"games_f.csv", basePath+"games_metadata_f.json")
      timeUtils.time(seqRecommender.recommend(targetUser), "Total time execution sequential", "Seq")

      val parRecommender = new parRecommendation(basePath+"recommendations_f.csv", basePath+"games_f.csv", basePath+"games_metadata_f.json")
      timeUtils.time(parRecommender.recommend(targetUser), "Total time execution parallel", "Par")



     /* dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath+"recommendations_f.csv")
      dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath+"games_f.csv")
      dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "games_metadata_f.json")

     val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MlLib")

      val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")

      val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL", "SQL")
      x
      */

    sparkLocal.stop()
  }
}