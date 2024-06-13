package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object recommendationSQL_RDD {
  def main(args: Array[String]): Unit = {

    //Initialize SparkSession
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val dataPathRec = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\games.csv"

    val tPreProcessingI = System.nanoTime()

    // Load dataset as DataFrame
    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    // Clean the dataset from useless whitespace and select useful columns
    val cleanMerge = dfRec.select("app_id", "user_id", "is_recommended")
      .join(dfGames.select("app_id", "title"), Seq("app_id"))
      .withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))
      .cache()

    // Tokenization of titles on whitespaces and aggregation by user ID
    val filteredData = cleanMerge
      .withColumn("words", split(col("title"), "\\s+"))
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))
      .filter(size(col("words")) >= 20)
      .cache()

    // Explode the aggregated data for TF-IDF calculation
    val explodedDF = filteredData.withColumn("word", explode(col("words"))).select("user_id", "word").cache()

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    // Calculate the total number of words associated with each unique user in the dataset
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    // Calculate the Term Frequency (TF) for each word and user combination
    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    // Calculate Document Frequency
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    // Counting the total number of users.
    val totalDocs = filteredData.count()

    // Calculate the IDF for each word in your RDD
    val idfRDD = dfDF.rdd.map { row =>
      val word = row.getString(0) // Assuming word is at index 0
      val docFreq = row.getLong(1) // Assuming document_frequency is at index 1
      val idf = math.log(totalDocs.toDouble / docFreq)
      (word, idf)
    }

    import spark.implicits._ // For toDF() method

    // Join the DataFrames on the 'word' column
    val tfidfDF = tf.join(idfRDD.toDF("word", "idf"), "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    val preprocessedDF = tfidfDF
      .groupBy("user_id")
      .agg(collect_list("word").alias("words"), collect_list("tf_idf").alias("tf_idf_values"))
      .withColumn("word_tfidf_map", map_from_arrays(col("words"), col("tf_idf_values")))

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()

    //Convert in a RDD[(Int, Map[String, Double])]
    val userSimRDD = preprocessedDF.rdd.map(row => {
      val userId = row.getAs[Int]("user_id")
      val wordTfidfMap = row.getAs[Map[String, Double]]("word_tfidf_map")
      (userId, wordTfidfMap)
    })

    // Input: two vectors as a map of words and weights
    // Output: cosine similarity
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
      def dotProduct(v1: Map[String, Double], v2: Map[String, Double]): Double = {
        v1.foldLeft(0.0) { case (acc, (key, value)) =>
          v2.get(key).map(value * _).getOrElse(0.0) + acc // Handle potential missing keys and type errors
        }
      }

      // Calculate vector magnitude (length)
      def magnitude(vector: Map[String, Double]): Double = {
        math.sqrt(vector.values.map(value => value * value).sum)
      }

      // Calculate cosine similarity
      dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
    }

    val targetUser = 2591067

    // Get users similar to the target
    def getSimilarUsers(userId: Int, tfidfValues: RDD[(Int, Map[String, Double])]): Array[(Int, Double)] = {
      val userGames = tfidfValues.lookup(userId).head
      tfidfValues.filter(_._1 != userId).map {
        case (otherUserId, otherUserGames) => (otherUserId, computeCosineSimilarity(userGames, otherUserGames))
      }.collect().sortBy(-_._2).take(3)
    }

    // Get recommendations for target users, based on previously calculated TF-IDF values
    val recommendedUsers = getSimilarUsers(targetUser, userSimRDD)

    println("recommendedUsers Top 3")
    recommendedUsers.foreach(println)

    /*
    (10941911,0.7293625797795579)
    (14044364,0.7263267622929318)
    (4509885,0.7186991307198306)
    */

    val tCosineSimilarityF = System.nanoTime()

    val tFinalRecommendI = System.nanoTime()

    // Extract games recommended by the target user
    val titlesPlayedByTargetUser = cleanMerge
      .filter(col("user_id") === targetUser)
      .select("title")
      .distinct()
      .as[String]
      .collect()

    // Extract relevant user IDs from recommendations
    val userIdsToFind = recommendedUsers.map(_._1).toSet

    // Filter datasetDF to remove already played games
    val finalRecommendations = cleanMerge
      .filter(col("user_id").isin(userIdsToFind.toSeq: _*) &&
        !col("title").isin(titlesPlayedByTargetUser: _*) &&
        col("is_recommended") === true)
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    val tFinalRecommendF = System.nanoTime()

    finalRecommendations.take(100).foreach(println)

    /*
    +-------+--------------------+----------+
    | app_id|               title|     users|
    +-------+--------------------+----------+
    |1085660|           destiny 2|[14044364]|
    |1172470|       apex legends™|[14044364]|
      | 307690|sleeping dogs: de...|[14044364]|
      |1267910|         melvor idle|[14044364]|
      |1227890|     summer memories| [4509885]|
      |1126290|                lost|[14044364]|
    |1153430|           love wish| [4509885]|
      |1109460|there is no greendam| [4509885]|
      |1146630|      yokai's secret| [4509885]|
      +-------+--------------------+----------+
  */
    // Calculating execution times
    println("\n\nExecution time(preprocessing):\t"+ (tPreProcessingF-tPreProcessingI)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (tTFIDFF-tTFIDFI)/1000000 + "ms\n")
    println("\n\nExecution time(Cosine similarity calculation):\t"+ (tCosineSimilarityF-tCosineSimilarityI)/1000000 + "ms\n")
    println("\n\nExecution time(final recommendation):\t"+ (tFinalRecommendF-tFinalRecommendI)/1000000 + "ms\n")
    println("\n\nExecution time(total):\t"+ (tFinalRecommendF-tPreProcessingI)/1000000 + "ms\n")

    spark.stop()
  }
}
