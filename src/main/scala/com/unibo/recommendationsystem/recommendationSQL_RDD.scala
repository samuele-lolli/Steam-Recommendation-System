package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object recommendationSQL_RDD {
  def main(args: Array[String]): Unit = {
    // Initialize SparkSession
    val spark = SparkSession.builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val dataPathRec = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\games.csv"

    val tPreProcessingI = System.nanoTime()

    // Load datasets
    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    // Clean and join datasets
    val cleanMerge = dfRec.select("app_id", "user_id", "is_recommended")
      .join(dfGames.select("app_id", "title"), Seq("app_id"))
      .withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))
      .cache()

    // Tokenize titles and aggregate by user ID
    val filteredData = cleanMerge
      .withColumn("words", split(col("title"), "\\s+"))
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))
      .filter(size(col("words")) >= 20)
      .cache()

    // Explode aggregated data for TF-IDF calculation
    val explodedDF = filteredData.withColumn("word", explode(col("words"))).select("user_id", "word").cache()

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    // Calculate total number of words for each user
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    // Calculate Term Frequency (TF)
    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    // Calculate Document Frequency (DF)
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    // Calculate total number of users
    val totalDocs = filteredData.count()

    // Calculate IDF for each word using SQL
    dfDF.createOrReplaceTempView("dfDF")
    val idfDF = spark.sql(s"""SELECT word, log($totalDocs / document_frequency) AS idf FROM dfDF""")

    // Join TF and IDF to get TF-IDF
    val tfidfDF = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    val preprocessedDF = tfidfDF
      .groupBy("user_id")
      .agg(collect_list("word").alias("words"), collect_list("tf_idf").alias("tf_idf_values"))
      .withColumn("word_tfidf_map", map_from_arrays(col("words"), col("tf_idf_values")))

    val tTFIDFF = System.nanoTime()
    val tCosineSimilarityI = System.nanoTime()

    // Convert to RDD[(Int, Map[String, Double])]
    val userSimRDD = preprocessedDF.rdd.map(row => {
      val userId = row.getAs[Int]("user_id")
      val wordTfidfMap = row.getAs[Map[String, Double]]("word_tfidf_map")
      (userId, wordTfidfMap)
    })

    // Function to compute cosine similarity
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
      def dotProduct(v1: Map[String, Double], v2: Map[String, Double]): Double = {
        v1.foldLeft(0.0) { case (acc, (key, value)) =>
          v2.get(key).map(value * _).getOrElse(0.0) + acc
        }
      }

      def magnitude(vector: Map[String, Double]): Double = {
        math.sqrt(vector.values.map(value => value * value).sum)
      }

      dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
    }

    val targetUser = 2591067

    // Get similar users to the target user
    def getSimilarUsers(userId: Int, tfidfValues: RDD[(Int, Map[String, Double])]): Array[(Int, Double)] = {
      val userGames = tfidfValues.lookup(userId).head
      tfidfValues.filter(_._1 != userId).map {
        case (otherUserId, otherUserGames) => (otherUserId, computeCosineSimilarity(userGames, otherUserGames))
      }.collect().sortBy(-_._2).take(10)
    }

    val recommendedUsers = getSimilarUsers(targetUser, userSimRDD)
    val tCosineSimilarityF = System.nanoTime()

    println("Top 3 similar users:")
    recommendedUsers.foreach(println)

    /*
    (10941911,0.7293625797795579)
    (14044364,0.7263267622929318)
    (4509885,0.7186991307198306)
    */

    val tFinalRecommendI = System.nanoTime()

    import spark.implicits._
    // Extract games recommended by the target user
    val titlesPlayedByTargetUser = cleanMerge
      .filter(col("user_id") === targetUser)
      .select("title")
      .distinct()
      .as[String]
      .collect()

    // Extract relevant user IDs from recommendations
    val userIdsToFind = recommendedUsers.take(3).map(_._1).toSet

    // Filter dataset to remove already played games and aggregate recommendations
    val finalRecommendations = cleanMerge
      .filter(col("user_id").isin(userIdsToFind.toSeq: _*)
        && !col("title").isin(titlesPlayedByTargetUser: _*)
        && col("is_recommended") === true)
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    val tFinalRecommendF = System.nanoTime()

    finalRecommendations.take(100).foreach(println)

    /*
    [1085660,destiny 2,WrappedArray(14044364)]
    [1172470,apex legends™,WrappedArray(14044364)]
    [307690,sleeping dogs: definitive edition,WrappedArray(14044364)]
    [1267910,melvor idle,WrappedArray(14044364)]
    [1227890,summer memories,WrappedArray(4509885)]
    [1126290,lost,WrappedArray(14044364)]
    [1153430,love wish,WrappedArray(4509885)]
    [1109460,there is no greendam,WrappedArray(4509885)]
    [1146630,yokai's secret,WrappedArray(4509885)]
    */

    // Calculate and display execution times
    println(s"\n\nExecution time (preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000} ms")
    println(s"\n\nExecution time (TF-IDF calculation): ${(tTFIDFF - tTFIDFI) / 1000000} ms")
    println(s"\n\nExecution time (cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000} ms")
    println(s"\n\nExecution time (final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000} ms")
    println(s"\n\nExecution time (total): ${(tFinalRecommendF - tPreProcessingI) / 1000000} ms")

    spark.stop()
  }
}

