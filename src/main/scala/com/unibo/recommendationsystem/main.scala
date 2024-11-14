package com.unibo.recommendationsystem
import com.unibo.recommendationsystem.utils.dataUtils.{createCustomDatasets, filterAppIds, filterUsersWithReviews, saveFilteredDataset}
import com.unibo.recommendationsystem.utils.schemaUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, concat_ws}
import org.apache.spark.sql.functions._

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
    sparkLocal.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")

    var basePath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/"

    var dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath + "recommendations.csv").filter("is_recommended = true")//.sample(withReplacement = false, fraction = 0.35)
    var dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath + "games.csv")
    var dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "games_metadata.json")
    val dfUsers = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.usersSchema).load(basePath + "users.csv")

    //LOCAL
    //4893896
    val targetUser = createCustomDatasets(sparkLocal, dfRec, dfGames, dfMetadata, dfUsers)
    if (targetUser != -1) {
      basePath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/filteredDataset/"
     // val parRecommender = new parRecommendation(basePath+"recommendations_f.csv", basePath+"games_f.csv", basePath+"games_metadata_f.json")
     // parRecommender.recommend(targetUser)

      //DISTRIBUTED


      dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath+"recommendations_f.csv").filter("is_recommended = true")
      dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath+"games_f.csv")
      dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath+"games_metadata_f.json")
     // val metaDataArray = dfMetadata.withColumn("tags", split(col("tags"), ","))

      val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      mllibRecommender.recommend(targetUser)

      //val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      //rddRecommender.recommend(targetUser)

      val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      sqlRecommender.recommend(targetUser)






    } else {
      println("Custom dataset creation skipped. Recommendations not generated.")


     // val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
     // mllibRecommender.recommend(targetUser = 4893896)

      val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
      rddRecommender.recommend(targetUser = 4893896)

    //  val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    //  sqlRecommender.recommend(targetUser = 4893896)

    }

    sparkLocal.stop()
  }
}