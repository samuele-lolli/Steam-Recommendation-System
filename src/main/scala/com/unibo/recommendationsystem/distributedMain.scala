package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.recommender.{mllibRecommendation, rddRecommendation, sqlRecommendation}
import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.SparkSession

object distributedMain {

  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder()
      .appName("RecommendationSystem")
      .config("spark.executor.instances", "7") // One executor for each node
      .config("spark.executor.cores", "3") // 3 cores for all executors
      .config("spark.executor.memory", "25g") // Executor memory allocation
      .config("spark.driver.cores", "4") // All cores available on driver
      .config("spark.driver.memory", "8g") // Driver memory allocation
      .config("spark.shuffle.compress", "true") // Compress data for shuffle
      .config("spark.shuffle.spill.compress", "true") // Compress data when shuffle spills
      .config("spark.network.timeout", "600s") // Timeout ten minutes
      .config("spark.sql.adaptive.enabled", "true") // Adaptive query execution
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") // Use Kryoserializer
      .config("spark.kryoserializer.buffer.max", "256m") // Kryo buffer
      .config("spark.local.dir", "/tmp/spark-temp") // Use SSD for temporary storage
      .config("spark.speculation", "true") // Enable speculative execution for slow tasks
      .getOrCreate()

    val basePath = "gs://dataproc-staging-us-central1-1020270449793-agano56l/data/"

    timeUtils.setLogFilePath(basePath+"result.txt")

    val targetUser = 4893896

    val dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath + "recommendations.csv").filter("is_recommended = true")//.sample(withReplacement = false, 0.24, 44)
    val dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath + "games.csv")
    val dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "games_metadata.json")

    /* Initialize and run the MLLIB recommender algorithm.*/
    val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MLLIB")

    /* Initialize and run the RDD-based recommender algorithm. */
    val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")

    /* Initialize and run the SQL-full-based recommender algorithm. */
    val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL_FULL", "SQL")
  }
}

