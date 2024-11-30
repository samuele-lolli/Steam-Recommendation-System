package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.recommender.{mllibRecommendation, rddRecommendation, sqlRecommendation, sqlRecommendationV2}
import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.SparkSession

object distributedMain {

  def main(args: Array[String]): Unit = {
    val sparkLocal = SparkSession.builder()
      .appName("RecommendationSystem")
      .config("spark.executor.instances", "7") // One executor per worker.
      .config("spark.executor.cores", "3") // Use 3 cores per executor.
      .config("spark.executor.memory", "25g") // ~85% of memory allocated for executors.
      .config("spark.driver.cores", "2") // Driver cores for master node.
      .config("spark.driver.memory", "8g") // Sufficient for query execution and small metadata broadcast.
      .config("spark.shuffle.compress", "true") // Compress shuffle data to optimize I/O.
      .config("spark.shuffle.spill.compress", "true") // Compress spilled shuffle data.
      .config("spark.network.timeout", "600s") // Extend timeout for long-running tasks.
      .config("spark.sql.adaptive.enabled", "true") // Enable adaptive query execution
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") // Use Kryo serializer for efficiency.
      .config("spark.kryoserializer.buffer.max", "256m") // Smaller buffer since data size is moderate.
      .config("spark.local.dir", "/tmp/spark-temp") // Use SSD for temporary storage.
      .config("spark.speculation", "true") // Enable speculative execution to handle stragglers.
      .getOrCreate()

    val basePath = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/"
    timeUtils.setLogFilePath(basePath+"result.txt")

    val targetUser = 4893896

    val dfRec = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath + "recommendations.csv").repartition(112).filter("is_recommended = true")//.sample(withReplacement = false, 0.50, 12345)
    val dfGames = sparkLocal.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath + "games.csv")
    val dfMetadata = sparkLocal.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "games_metadata.json")

    val mllibRecommender = new mllibRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    timeUtils.time(mllibRecommender.recommend(targetUser), "Total time execution MlLib", "MlLib")

    /* Initialize and run the RDD-based recommender algorithm. */
    val rddRecommender = new rddRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    timeUtils.time(rddRecommender.recommend(targetUser), "Total time execution RDD", "RDD")

    /* Initialize and run the SQL-hybrid-based recommender algorithm. */
    val sqlRecommenderV2 = new sqlRecommendationV2(sparkLocal, dfRec, dfGames, dfMetadata)
    timeUtils.time(sqlRecommenderV2.recommend(targetUser), "Total time execution SQL_HYBRID", "SQL_HYBRID")

    /* Initialize and run the SQL-full-based recommender algorithm. */
    val sqlRecommender = new sqlRecommendation(sparkLocal, dfRec, dfGames, dfMetadata)
    timeUtils.time(sqlRecommender.recommend(targetUser), "Total time execution SQL_FULL", "SQL_FULL")


}
}

